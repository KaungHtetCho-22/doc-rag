# DocRAG Evaluation Report

**Date:** 2026-03-05 02:24  
**Mode:** GROQ  
**Questions:** 20  

## Ragas Scores

| Metric | Score | Bar | Target |
|--------|-------|-----|--------|
| Faithfulness     | 0.420 | ████████░░░░░░░░░░░░ | > 0.80 |
| Answer Relevancy | 0.795 | ███████████████░░░░░ | > 0.75 |
| Context Recall   | 0.280 | █████░░░░░░░░░░░░░░░ | > 0.70 |
| Context Precision| 0.425 | ████████░░░░░░░░░░░░ | > 0.70 |

## Target Assessment

- Faithfulness: 0.420 ❌ FAIL (target: 0.8)
- Answer Relevancy: 0.795 ✅ PASS (target: 0.75)
- Context Recall: 0.280 ❌ FAIL (target: 0.7)
- Context Precision: 0.425 ❌ FAIL (target: 0.7)

**Overall: ⚠️ Some targets missed**

## Per-Question Breakdown

| # | Question | Faith | Relev | Recall | Prec |
|---|----------|-------|-------|--------|------|
| 1 | How do transformer architectures improve obje... | 0.20 | 0.80 | 0.00 | 0.40 |
| 2 | What is the role of self-attention in visual ... | 0.20 | 0.80 | 0.40 | 0.40 |
| 3 | How does contrastive learning work for visual... | 0.80 | 1.00 | 0.80 | 0.60 |
| 4 | What are the main advantages of anchor-free o... | 0.20 | 0.80 | 0.00 | 0.40 |
| 5 | How do vision transformers handle variable im... | 1.00 | 1.00 | 0.20 | 0.00 |
| 6 | What loss functions are commonly used for tra... | 0.00 | 0.00 | 0.40 | 0.20 |
| 7 | How is non-maximum suppression used in object... | 0.80 | 0.90 | 0.20 | 0.60 |
| 8 | What is the difference between instance segme... | 0.80 | 1.00 | 1.00 | 0.70 |
| 9 | How do feature pyramid networks improve multi... | 0.20 | 0.80 | 0.40 | 0.60 |
| 10 | What techniques are used to reduce computatio... | 0.20 | 0.60 | 0.00 | 0.20 |
| 11 | What datasets are commonly used to evaluate o... | 0.40 | 1.00 | 0.80 | 0.60 |
| 12 | How is mean average precision calculated for ... | 0.00 | 1.00 | 0.00 | 0.20 |
| 13 | What are the main challenges in 3D point clou... | 0.00 | 1.00 | 0.00 | 0.20 |
| 14 | How do self-supervised methods compare to sup... | 0.60 | 0.80 | 0.00 | 0.40 |
| 15 | What metrics are used to evaluate video objec... | 0.80 | 0.60 | 0.00 | 0.40 |
| 16 | How are transformer decoders used in end-to-e... | 0.60 | 1.00 | 0.00 | 0.80 |
| 17 | What is deformable attention and why is it us... | 0.80 | 1.00 | 0.80 | 0.80 |
| 18 | How does knowledge distillation improve small... | 0.20 | 0.80 | 0.00 | 0.20 |
| 19 | What role does data augmentation play in trai... | 0.60 | 1.00 | 0.60 | 0.80 |
| 20 | How are 2D and 3D detection approaches combin... | 0.00 | 0.00 | 0.00 | 0.00 |

## Latency

- Avg: 3484ms
- P95: 8016ms
- Min: 376ms
- Max: 8016ms