# QK-Clip

Muon optimizer와 함께 사용되는 post-update weight clipping 기법으로, attention score explosion을 방지한다.
Moonshot AI가 Kimi K2 학습에서 제안하고 검증했다.

## 배경: 왜 필요한가

Muon optimizer는 AdamW 대비 token efficiency가 높지만, large scale (수십B+ 파라미터)에서 attention logit이 폭발하는 문제가 있다. Softmax 입력값이 1000+까지 치솟으면 NaN/loss spike가 발생하고 학습이 crash된다.

기존 해결책인 QK-Norm (RMSNorm을 Q, K에 적용)은 forward pass를 변경하므로 MLA 등과의 호환성 문제가 있다. QK-Clip은 **forward/backward를 건드리지 않고** optimizer step 이후에 weight만 rescale하는 방식이라 범용성이 높다.

## 알고리즘

### 1. Max Logit 계산

배치 B에 대해 각 attention head h의 max attention logit을 계산한다:

```
S_max^h = (1/√d) * max_{X∈B, i,j} |q_i^h · k_j^h|
```

### 2. Scaling Factor

Threshold τ (보통 100)를 기준으로 per-head scaling factor를 구한다:

```
γ_h = min(1, τ / S_max^h)
```

- `S_max^h ≤ τ`이면 `γ_h = 1` → clipping 안 함
- `S_max^h > τ`이면 `γ_h < 1` → clipping 발동

### 3. Weight Rescaling

**Standard MHA의 경우:**

```
W_q^h ← W_q^h * γ_h^α
W_k^h ← W_k^h * γ_h^(1-α)
```

여기서 α ≈ 0.5로 Q와 K에 균등하게 분배한다.
`γ_h^α * γ_h^(1-α) = γ_h`이므로 `q·k`에 대한 총 scaling은 정확히 γ_h이다.

**MLA (Multi-head Latent Attention)의 경우:**

MLA에서는 Q, K가 shared/unshared component로 분해된다:
- **head-specific context components (qc, kc):** `√γ_h`로 scaling
- **head-specific rotary components (qr):** `γ_h`로 scaling
- **shared rotary (kr):** scaling 안 함 (cross-head 영향 방지)

## 핵심 특성

| 특성 | 설명 |
|------|------|
| **Per-head** | 전체 layer가 아닌 head 단위로 clip. 문제 있는 head만 개입 |
| **Post-update** | Optimizer step 이후에 적용. Forward/backward 변경 없음 |
| **Self-deactivating** | 학습 초기(~70k steps)에만 활발히 작동, 이후 자연스럽게 비활성화 |
| **Forward-pass 불변** | QK-Norm과 달리 inference 동작 변경 없음, MLA 호환 |

## Pseudocode

```python
def qk_clip(model, threshold=100.0, alpha=0.5):
    """Optimizer step 이후에 호출"""
    max_logit = 0.0

    for layer in model.attention_layers:
        for h in range(num_heads):
            # 현재 step의 batch에서 이미 계산된 max logit 사용
            s_max = layer.cached_max_logit[h]
            max_logit = max(max_logit, s_max)

            gamma = min(1.0, threshold / s_max)
            if gamma < 1.0:
                layer.W_q[h] *= gamma ** alpha
                layer.W_k[h] *= gamma ** (1 - alpha)

    return max_logit
```

## 학습 결과 (Kimi K2)

- **모델:** 1T total params, MoE, 15.5T tokens
- **Threshold:** τ = 100
- **초기 (~70k steps):** QK-Clip이 활발히 작동하며 max logit을 100 이하로 유지
- **후기:** Max logit이 자연스럽게 ~30 수준으로 안정화, clipping 불필요
- **결과:** 전체 학습 중 loss spike 0회

Mid-scale 검증 (9B activated / 53B total MoE)에서도 vanilla Muon은 max logit 1000+ 폭발, QK-Clip 적용 시 안정화를 확인했다.

## Flash Attention에서의 구현 고려사항

QK-Clip 자체는 optimizer-side 로직이라 flash attention kernel 내부를 수정할 필요는 없다. 다만:

1. **Max logit 통계 수집:** Forward pass에서 per-head max attention score를 추출해야 한다. 현재 FA4는 LSE (log-sum-exp)를 반환하지만, raw max logit은 별도 구현이 필요할 수 있다.

2. **Softcap과의 관계:** `softcap`은 forward pass에서 attention score를 tanh로 clamp하는 반면, QK-Clip은 weight를 직접 조정하므로 상호 배타적이지 않지만 목적이 겹친다.

3. **Score mod 활용 가능성:** FA4의 `score_mod` 인터페이스로 attention score monitoring을 커스텀 구현할 수 있을 가능성이 있다.

## References

- Kimi K2 Technical Report (arXiv:2507.20534)
- Megatron-Core `core.optimizer.qk_clip` 구현
- Muon Optimizer (github.com/KellerJordan/Muon)
