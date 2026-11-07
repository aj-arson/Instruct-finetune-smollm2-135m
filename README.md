# Instruction-tuning SmolLM2-135M from scratch (a learning project)

Full fine-tuning of the `HuggingFaceTB/SmolLM2-135M` **base** model into an instruction-following model, trained on a single Kaggle GPU in fp32. The goal was to learn how instruction tuning works end to end (data, loss masking, overfitting, evaluation), not to beat a benchmark.


## Wanna try it?

The application is deployed using streamlit. You can try the deployed application using below link.
Follow this link: https://instruct-finetune-smollm2-135m-not-chagpt.streamlit.app/

## Summary

| Question | Answer |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M` (base, not instruct) |
| Method | Full fine-tuning, fp32, custom PyTorch training loop |
| Hardware limits | Kaggle GPU, max sequence length 1024, batch size 4, 12 h per session |
| Final model | Alpaca-cleaned SFT, 2 epochs |
| Main finding | A bigger dataset with fewer epochs fixed the overfitting, with no extra regularization beyond the default dropout |

## Experiments

### Run 1: Dolly-15k, 5 epochs (overfit)
- Data: `databricks/databricks-dolly-15k`, 12,008 training examples.
- Eval loss was lowest early (about 2.09 at the end of epoch 0) and then rose every epoch to about 2.43, while train loss fell from about 1.75 to about 0.86. Classic overfitting.
- Outputs: followed the format but copied source text, hallucinated facts, and kept generating after the end token.

### Regularization attempts
- Attention dropout of 0.1 did not resolve the overfitting.
- Changing `rms_norm_eps` was considered and rejected: it is a numerical-stability constant, not a regularizer.

### Run 2: Alpaca-cleaned, 2 epochs (final)
- Data: `yahma/alpaca-cleaned` (51,760 examples).
- Run 3: 20% held out for eval (about 10K). Final train loss about 1.19, eval loss 1.4767.
- Run 4: 5% held out for eval (about 3K). Final train loss about 1.18, eval loss 1.4646.
- Eval loss did not rise in either run. The train/eval gap widened mildly in epoch 2, and the gains in epoch 2 were tiny, so two epochs was the right stopping point.
- Only the dataset size and number of epochs changed. No regularization beyond the default dropout was added. (This is not a controlled ablation, since both changed together.)

## Training setup
- Chat format: ChatML-style turns ending with `<|im_end|>`, appended exactly once per assistant turn.
- Loss computed on response tokens only (prompt tokens masked with `-100`).

## Evaluation
A 50-prompt benchmark (`eval_prompts.csv`) written for this project, covering six categories: classification, list generation, extraction and rewriting, short Q&A, open-ended writing, and multi-constraint instructions. Outputs were generated with greedy decoding and scored on a rubric by an automated judge:

- follows the instruction (0/1)
- respects format and constraints (0/1)
- stops cleanly (0/1)
- content correctness (0 to 2)

Total per prompt is out of 5, so 250 across the benchmark.

### Results

| Metric (average) | Alpaca (ours) | Dolly (ours) | `SmolLM2-135M-Instruct` |
|---|---|---|---|
| Follows the instruction | 70% | 50% | 64% |
| Respects format and constraints | 66% | 52% | 54% |
| Stops cleanly | 94% | 70% | 94% |
| Content correctness (0 to 2) | 1.12 | 0.82 | 0.86 |
| **Total (of 250)** | **171** | **127** | **149** |

### Prompt-overlap check
Prompts were fuzzy-matched against Alpaca-cleaned. Three were close to the same question (South America countries, Romeo and Juliet, boiling point at sea level), and about eight more were the same type of generic task. Removing the flagged prompts left the ranking and the size of the gap unchanged: on the flagged prompts the Alpaca and official models were nearly tied, and the gap came from the other prompts.

## Takeaways
1. **Data size and epochs mattered most.** Going from 12k examples for 5 epochs to about 50k for 2 epochs removed the overfitting without extra regularization.
2. **Judge by outputs, not only loss.** Eval losses from different datasets are not comparable, so a fixed prompt set with side-by-side scoring was the real exit criterion.
3. **Our Alpaca model is roughly on par with the official instruct model on this benchmark.** The benchmark is small (50 prompts, one greedy generation each, automated scoring) and written in the short-instruction style of the training data, so treat the lead as "about on par", not a win.
4. **Dolly-15k was clearly the weakest data source.** It produced more loops and missing stop tokens and ignored edit-style instructions.
5. **Remaining weaknesses are mostly size limits:** exact constraints ("exactly two sentences", "only yes or no"), multi-item classification, factual recall, and occasional repetition loops. A `repetition_penalty` of about 1.1 at inference is a cheap mitigation.

## Pitfalls we hit
- Doubled `<|im_end|>` followed by junk text: the end token was appended twice (template plus code), and generation did not stop on it. Fix: append it once, and stop with `eos_token_id` set to the `<|im_end|>` token id (or pass `tokenizer=` when using `stop_strings`).
- Newer `transformers` returns a `BatchEncoding` from `apply_chat_template`, so use `return_dict=True` and call `model.generate(**inputs, ...)`.
- In the comparison script, a later `model = load_model_and_tokenizer(...)` call silently overwrote the Hugging Face model, so the first "official model" results were really from our own model. Print `model.config._name_or_path` before a run.

## Limitations
- Small benchmark, single judge, single greedy generation per prompt.
- Overlap check catches similar wording only, not paraphrases.
- Licenses: check the dataset licenses before releasing a model (Dolly is CC-BY-SA, Alpaca-cleaned is non-commercial as far as I know; verify both).

## Next: tool use
Continue from the Alpaca checkpoint and teach function calling with the same next-token objective:
- Data: a subset of `Salesforce/xlam-function-calling-60k` converted to our chat template, plus synthetic no-tool and missing-tool examples.
- Loss only on assistant turns (tool call and final answer), masked everywhere else.
- Evaluate with valid-format rate, tool-selection accuracy, argument accuracy, and no-call accuracy, with some tools held out of training.

## Files
- `eval_prompts.csv`: the 50-prompt benchmark
- `scored_outputs.csv`, `scored_outputs_dolly.csv`, `scored_outputs_hf_instruct.csv`: per-prompt scores for each model
- `comparison_three_models.csv`: side-by-side totals and notes