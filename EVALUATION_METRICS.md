# Evaluation Metrics (for verification)

| # | Metric | Definition | How we measure it now | What it shows |
|---|---|---|---|---|
| 1 | Under-triage rate | % of truly EMERGENCY/SEVERE cases labelled lower or left INCOMPLETE | Test cases built from dataset disease profiles with only 2–4 symptoms kept (mimics a short caller message); true label = that disease's severity in `disease_severity.csv` | Safety: urgent patients are not missed |
| 2 | Over-triage rate | % of truly MILD/MODERATE cases escalated to SEVERE/EMERGENCY | Same test cases | Cost of unnecessary rural referrals |
| 3 | Silent-error vs abstention rate | % of answered cases with the wrong label; % of cases where the system asks/abstains instead | Same test cases | Core thesis: uncertainty triggers a question, not a guess |
| 4a | Follow-up resolution rate | % of uncertain cases ending with the correct label and a single remaining candidate | Scripted caller answers "yes" if the asked symptom is in the true profile, otherwise "no" | Follow-up turns uncertainty into a correct decision |
| 4b | Mean questions asked | Average follow-up questions per uncertain case (max 3) | Same simulation | Burden on the caller |
| 4c | Reply-not-understood rate | % of caller replies the system cannot parse as yes/no | Simulated replies include realistic variants ("haan", "nahi", "I think so", "maybe") | Works with how real callers reply |
| 5 | Routing correctness | % of urgent cases sent to the best eligible facility | Compare each route against a brute-force best choice over the demo facility list | Right care at the right place |
| 6 | Decision consistency | % of cases where the same message (repeated or paraphrased) gets the same label | Run the full pipeline (Ollama + FAISS + classifier) several times per free-text message | Reliability of the LLM-based pipeline |

**Limitation:** true labels come from the dataset, not from clinicians.
