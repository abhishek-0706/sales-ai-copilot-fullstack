# Fine-tuning the retrieval embedder (bge-small-en-v1.5)

**What:** the open-source embedding model `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions, MIT licence) was fine-tuned on synthetic sales questions about this project's CRM corpus, on a laptop CPU, then exported to ONNX so the app can use it through fastembed without torch at runtime (`EMBEDDER=finetuned`). The default embedder is unchanged (`hash`).

**Why:** the existing golden set could not show whether a better embedder helps: base bge-small already scores 1.0 vector hit@6 on it. A larger, harder, held-out test set was written first, and both models were measured on it with the app's own retrieval code.

## Result in one paragraph

The fine-tune worked mechanically: on held-out *synthetic* validation questions, vector MRR@10 rose from 0.751 to 0.956. On the **hand-written held-out test set it made no difference beyond noise.** Vector-only retrieval improved slightly (MRR@10 +0.021 filtered, +0.035 unfiltered). Hybrid retrieval, which the app uses, got slightly worse (MRR@10 -0.024 and -0.017). Every 95% confidence interval includes zero, and the rank of the first correct chunk improved for about as many questions as it worsened. The golden evaluation gate still passes with zero access leakage. The likely reason for the gap: the synthetic questions share 65% of their content words with the answer passage, the hand-written ones 22%, so the model learned an easier task than the one tested. A truthful claim is "fine-tuned bge-small on CPU and measured no significant gain on a held-out paraphrase test", not "improved retrieval".

## Results (run on 2026-11-24)

### Frozen test set, 68 hand-written questions (hit@1 / hit@3 / hit@6 / MRR@10)

| Setting / mode | hash (default) | base bge-small | fine-tuned | fine-tuned minus base: MRR@10 (95% CI) | rank better / worse |
|---|---|---|---|---|---|
| filtered / vector | .397 / .618 / .691 / .520 | .706 / .912 / .985 / .817 | .721 / .956 / .985 / .838 | +0.021 (-0.061, +0.107) | 14 / 13 |
| filtered / **hybrid** (app default) | .544 / .779 / .912 / .685 | .779 / .956 / 1.000 / .867 | .735 / .956 / 1.000 / .844 | -0.024 (-0.081, +0.032) | 6 / 9 |
| unfiltered / vector | .294 / .471 / .588 / .403 | .647 / .882 / .956 / .770 | .676 / .941 / .985 / .804 | +0.035 (-0.051, +0.124) | 14 / 12 |
| unfiltered / hybrid | .529 / .618 / .765 / .609 | .750 / .926 / 1.000 / .846 | .721 / .971 / .985 / .829 | -0.017 (-0.082, +0.050) | 7 / 11 |

Keyword-only (BM25) does not depend on the embedder: .647 / .824 / .926 / .751 filtered, .500 / .721 / .882 / .642 unfiltered.

- "base bge-small" is the base model exported through the same ONNX pipeline as the fine-tuned one. The app's default `EMBEDDER=fastembed` (Qdrant's quantized ONNX build) gave **identical** numbers on every row, so quantization plays no part in the comparison.
- Confidence intervals are from 10,000 paired bootstrap resamples over questions (`results/compare_test_base-onnx_vs_finetuned.json`). Exact sign tests on the questions whose rank changed give p = 0.48 to 1.0. With 68 questions, one question moves hit@k by 1.5 points.
- hit@k intervals are just as wide. For example, unfiltered vector hit@3 is +0.059 (-0.029, +0.147). Unfiltered hybrid hit@6 dropped from 1.000 to 0.985 (one question fell from the top 6 to rank 7 or below).
- The only sizeable movement is on the 11 product-documentation questions in vector mode (MRR@10 0.579 to 0.727). That is too few questions to claim anything.

### Overfitting check (unfiltered, through the app's retrieval code)

| Questions | n | vector hit@1 base -> fine-tuned | vector MRR@10 base -> fine-tuned | hybrid MRR@10 base -> fine-tuned | BM25 MRR@10 |
|---|---|---|---|---|---|
| training (synthetic) | 796 | 0.678 -> 0.959 | 0.788 -> 0.976 | 0.887 -> 0.951 | 0.896 |
| validation (synthetic, held out) | 88 | 0.636 -> 0.920 | 0.751 -> 0.956 | 0.858 -> 0.931 | 0.877 |
| test (hand-written, held out) | 68 | 0.647 -> 0.676 | 0.770 -> 0.804 | 0.846 -> 0.829 | 0.642 |

Training and validation are close, so the model did not memorise its training questions. The gap is between synthetic and hand-written questions. BM25 alone already reaches MRR 0.88 on the synthetic questions but 0.64 on the test questions, and the mean share of a question's content words that appear in its answer passage is 0.65 (training), 0.65 (validation) and 0.22 (test). The LLM was asked for synonyms and indirect wording but mostly reused the records' words.

### Regression check: golden evaluation with `EMBEDDER=finetuned`

`python -m eval.run_eval --gate` **passes**, and every retrieval and answer metric is identical to base bge-small (`eval/results/offline_finetuned_bge-small-en-v1.5-sales-ft.json`):

- evidence hit rate 0.963;
- retrieval hit@6 vector / keyword / hybrid 1.0 / 0.926 / 0.963, paraphrased subset 1.0 / 0.6 / 0.8;
- citation hit 0.815, citation validity 1.0, answer fact rate 0.815;
- **access leakage 0**, negative-case violations 0, injection reaching the model 0, PII values in prompts 0.

The only change is that the planted injection note was retrieved, and quarantined, 19 times instead of 20.

### Not completed: held-out-account run

A second run that leaves three accounts (ACC-001, ACC-002, ACC-011, chosen by seed before any results) out of training would show whether the fine-tune transfers to documents it never saw. `train.py --holdout-accounts` supports it. It was stopped at step 70 of 300, when the shared CPU slowed it to about 30 s per step (two more hours). Only 16 test questions concern those accounts, so the result could not have been conclusive anyway: the 95% interval would be about ±0.15 MRR.

## Method

### 1. Held-out test set (written first, then frozen)

`data/test_queries.json`: **68 questions written by hand** from `backend/data` before any training data existed. They use synonyms and indirect, salesperson-style phrasing and avoid the target text's keywords where possible ("Which freight customer has people keying in tens of thousands of transport papers by hand each month?" for a note that says "manual processing of about 40,000 shipping documents per month"). 54 describe the account without naming it, 3 name it, 11 ask about product documentation. Each question lists every chunk that states the answer (85 distinct chunks of the 115 in the index are targets). The file's SHA-256 is in `data/test_queries.sha256`; `eval_retrieval.py` refuses to run if the file changes. The baseline was measured before any training.

Two settings are reported:

- **filtered**: the same as `eval/run_eval.py`. Each question is asked as a persona and the access predicate is applied inside the search (personas see 38 to 75 of the 115 chunks). `t-ae-na` is a test-only AE for NA-East, because no user in `users.json` covers that territory.
- **unfiltered**: an all-access persona sees all 115 chunks, so there are more distractors. This is the harder retrieval test.

Metrics: hit@k (a relevant chunk in the top k; the app passes the top 6 to the model) and MRR@10 (mean of 1/rank of the first relevant chunk, 0 if outside the top 10). Modes: `vector` (embedder only), `hybrid` (vector + BM25 with reciprocal rank fusion, which the app uses), `keyword` (BM25 only; does not depend on the embedder).

### 2. Training data (synthetic, generated differently from the test set)

`make_training_data.py` (app venv):

- **LLM questions**: `openai/gpt-oss-120b` on Groq's free tier (any OpenAI-compatible endpoint works) received the records of one account at a time (or half of the product docs) and wrote questions in mixed styles, labelling every record id that answers each one. It never saw the test set. The raw output is cached in `data/llm_questions.jsonl`, so the later steps rerun without an API key.
- **Template questions**: short keyword-style questions per account and record type ("Granmoor Insurance deal risk", "Torvane close date"), with answers known from the record type.
- **Filters** (all reported in `data/leakage_report.json`):
  - ambiguity: LLM questions that contain no word stem unique to their account ("How many employees does the client have?") cannot be answered from the question alone and were dropped;
  - string near-duplicates of a test or golden-eval question (difflib ratio >= 0.7 or content-word Jaccard >= 0.5) were dropped;
  - semantic paraphrases of a test question (base-model cosine >= 0.8) that share its answer chunk were dropped. The string check alone let through, for example, "How long is the call handling time that the bank wants to improve?" (cosine 0.886 to test question t23).
- **Split**: 10% of questions held out as a validation set (seed 42). The test set was evaluated once per trained model, with no tuning afterwards. For full disclosure: a first training run was stopped after 4 steps, when the semantic-paraphrase filter was added and the CPU settings were reduced; nothing from it was evaluated.

| Step | Questions |
|---|---|
| Generated: LLM / template | 1,115 / 276 |
| After removing exact duplicates | 1,335 |
| Dropped: no account cue (ambiguous) | 420 |
| Dropped: string near-duplicate of a test or golden question | 31 (2 of them nearest to a test question) |
| Dropped: semantic paraphrase of a test question with the same answer | 15 |
| **Training** | **796 questions, 1,141 (question, passage) pairs** |
| **Validation** | **88 questions, 128 pairs** |

Leakage checks on what remains: no exact test question. Highest difflib ratio to any test or golden question 0.694, highest content-word Jaccard 0.444. Highest base-model cosine to a test question with the same answer 0.799; 32 kept questions score 0.75 to 0.8 against a same-answer test question, because training covers the same facts as the test, in other words. Full lists are in `data/leakage_report.json`.

### 3. Training

`train.py` (training venv, CPU): sentence-transformers `MultipleNegativesRankingLoss` (contrastive loss: each question must score its positive passage above every other passage in the batch) with one **hard negative** per pair, mined with the base model: the most similar passage that is not a labelled answer and does not repeat a clause of one (the generated emails and meetings restate activity notes, so a naive miner picks correct answers as negatives). A custom batch sampler puts at most one question per account or product in a batch, so another account's passage in the batch is a true negative. Hyperparameters were fixed before any test evaluation: 2 epochs, batch size up to 16 (the one-per-account rule makes batches average about 7.6 rows, 150 steps per epoch), learning rate 3e-5, 10% warmup, max sequence length 128 (every passage fits), seed 42. The hard-negative run took **41.7 minutes** for 300 steps on 4 threads of an Intel Core Ultra 7 165U shared with another CPU-heavy job (torch 2.15.0+cpu, sentence-transformers 6.3.0, Python 3.14). Training loss fell from 1.45 to about 0.1 by the end of epoch 1 and to 0.001 at the last step; validation MRR@10 was 0.920 after epoch 1 and 0.956 after epoch 2 (`results/train_sales-ft.json`).

### 4. Export and use in the app

`export_onnx.py` exports the transformer to ONNX (last hidden state; fastembed applies CLS pooling and L2 normalisation, as bge does) with the tokenizer files, and checks the ONNX output against torch. `app/embeddings.py` `FineTunedEmbedder` registers the directory as a fastembed custom model; `EMBEDDER=finetuned` loads `backend/.models/bge-small-en-v1.5-sales-ft/`. If the files are missing it raises an error that says how to build them rather than silently using another model. The base model is exported the same way (`bge-small-en-v1.5-base-onnx`) as a control, because the app's default `fastembed` model is Qdrant's quantized ONNX build: comparing fine-tuned fp32 against base fp32 isolates the effect of training from the effect of quantization.

## Reproduce

```powershell
cd backend
# training venv (torch CPU etc.), kept separate so the app's requirements stay lean
python -m venv finetune\.venv-train
finetune\.venv-train\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
finetune\.venv-train\Scripts\python -m pip install -r finetune\requirements-train.txt

# 1. baseline on the frozen test set (app venv)
..\.venv\Scripts\python -m finetune.eval_retrieval --embedders hash,fastembed
# 2. training data (calls the LLM only if data/llm_questions.jsonl is missing or with --regenerate;
#    needs LLM_BASE_URL, LLM_MODEL, LLM_API_KEY in the environment; the paraphrase check needs fastembed,
#    from requirements-local-embeddings.txt)
..\.venv\Scripts\python -m finetune.make_training_data
# 3. train, then export fine-tuned and base models to ONNX
finetune\.venv-train\Scripts\python finetune\train.py --run sales-ft
finetune\.venv-train\Scripts\python finetune\export_onnx.py --model .models\st\sales-ft\final --out .models\bge-small-en-v1.5-sales-ft
finetune\.venv-train\Scripts\python finetune\export_onnx.py --model BAAI/bge-small-en-v1.5 --out .models\bge-small-en-v1.5-base-onnx
# 4. evaluate and compare (paired bootstrap CI and sign test)
..\.venv\Scripts\python -m finetune.eval_retrieval --embedders hash,fastembed,base-onnx,finetuned
..\.venv\Scripts\python -m finetune.eval_retrieval --compare base-onnx finetuned
# overfitting check: same metrics on the training and validation questions
..\.venv\Scripts\python -m finetune.eval_retrieval --embedders base-onnx,finetuned --queries finetune\data\val.jsonl --split val
..\.venv\Scripts\python -m finetune.eval_retrieval --embedders base-onnx,finetuned --queries finetune\data\train.jsonl --split train
# optional, NOT completed here: held-out-account run (train without three accounts, score test questions about them)
finetune\.venv-train\Scripts\python finetune\train.py --run sales-ft-holdout --holdout-accounts ACC-001,ACC-002,ACC-011
finetune\.venv-train\Scripts\python finetune\export_onnx.py --model .models\st\sales-ft-holdout\final --out .models\bge-small-en-v1.5-sales-ft-holdout
..\.venv\Scripts\python -m finetune.eval_retrieval --embedders finetuned-holdout
..\.venv\Scripts\python -m finetune.eval_retrieval --compare base-onnx finetuned-holdout --accounts ACC-001,ACC-002,ACC-011
# 5. regression check on the golden eval with the fine-tuned model
$env:EMBEDDER="finetuned"; ..\.venv\Scripts\python -m eval.run_eval --gate; Remove-Item Env:EMBEDDER
```

## Limitations and honest caveats

- **Synthetic everything.** The corpus is 115 generated chunks for 12 fictional accounts, built from templates, so records repeat each other's wording. The training questions are LLM- and template-generated. Real CRM text is messier and more varied; a gain here says nothing about a gain on real data.
- **Small test set, one author.** 68 questions, written by one author (the AI assistant that built this pipeline) before any model was run on them. One question is 1.5 percentage points. Confidence intervals are in the results; treat differences inside them as noise.
- **The model was trained on the same documents it is tested on** (with different questions). That is the normal setup for adapting a retriever to a known corpus, but it measures adaptation to this corpus, not general retrieval quality. Even with that advantage the test gain was within noise. Transfer to unseen documents was not measured (see "Not completed").
- **Ceiling effects.** Base bge-small already puts a correct chunk in the top 6 for nearly every test question, especially with BM25 fusion, so hit@6 cannot improve much. hit@1 and MRR are the informative metrics.
- **One seed, one hyperparameter setting**, no search. A different seed would move the numbers by a few points.
- **Compute.** Training ran on a shared 15 W laptop CPU (Intel Core Ultra 7 165U) while another job saturated most cores, so wall-clock times are pessimistic.
- **Not done:** the held-out-account run (stopped, see above), an in-batch-negatives-only ablation (`--no-hard-negatives` exists but was not run), multiple seeds, a quantized export of the fine-tuned model, and any test on real CRM data.
- **What would likely help next:** training questions with genuinely low word overlap (for example, generating them from a paraphrase of the record rather than from the record, or rejecting questions whose content words mostly occur in the passage), a larger hand-written test set, and a reranker for the hybrid fusion problem noted in the main README.
