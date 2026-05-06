CLASSIFICATION_PROMPT = """You are an expert in interpreting how LLMs solve math problems
using multi-step reasoning. Your task is to analyze a
chain-of-thought reasoning trace, broken into discrete text
sentences, and label each sentence with:
1. **function_tags**: One or more labels that describe what this
sentence is *doing* functionally in the reasoning process.
2. **depends_on**: A list of earlier sentence indices that this
sentence directly depends on, e.g., uses information, results, or
logic introduced in earlier sentences.
This annotation will be used to build a dependency graph and
perform causal analysis, so please be precise and conservative:
only mark a sentence as dependent on another if its reasoning
clearly uses a previous sentence’s result or idea.
Function Tags:
1. problem_setup: Parsing or rephrasing the problem (initial
reading or comprehension).
2. plan_generation: Stating or deciding on a plan of action
(often meta-reasoning).
3. fact_retrieval: Recalling facts, formulas, problem details
(without immediate computation).
4. active_computation: Performing algebra, calculations,
manipulations toward the answer.
5. result_consolidation: Aggregating intermediate results,
summarizing, or preparing final answer.
6. uncertainty_management: Expressing confusion, re-evaluating,
proposing alternative plans (includes backtracking).
7. final_answer_emission: Explicit statement of the final boxed
answer or earlier sentences that contain the final answer.
8. self_checking: Verifying previous steps, checking
calculations, and re-confirmations.
9. unknown: Use only if the sentence does not fit any of the
above tags or is purely stylistic or semantic.
Dependencies:
For each sentence, include a list of earlier sentence indices that
the reasoning in this sentence *uses*. For example:
- If sentence 9 performs a computation based on a plan in sentence
4 and a recalled rule in sentence 5, then depends_on: [4, 5]
- If sentence 24 plugs in a final answer to verify correctness
from sentence 23, then depends_on: [23]
- If there’s no clear dependency use an empty list: []
- If sentence 13 performs a computation based on information in
sentence 11, which in turn uses information from sentence 7, then
depends_on: [11, 7]
Important Notes:
- Make sure to include all dependencies for each sentence.
- Include both long-range and short-range dependencies.
- Do NOT forget about long-range dependencies.
- Try to be as comprehensive as possible.
- Make sure there is a path from earlier sentences to the final
answer.
Output Format:
Return a dictionary with one entry per sentence, where each entry
has:
- the sentence index (as the key, converted to a string),
- a dictionary with:
- "function_tags": list of tag strings
- "depends_on": list of sentence indices, converted to strings
Here is the expected format:
{
"1": {
"function_tags": ["problem_setup"],
"depends_on": [""]
},
"4": {
"function_tags": ["plan_generation"],
"depends_on": ["3"]
},
"5": {
"function_tags": ["fact_retrieval"],
"depends_on": []
},
"9": {
"function_tags": ["active_computation"],
"depends_on": ["4", "5"]
},
"24": {
"function_tags": ["uncertainty_management"],
"depends_on": ["23"]
},
"32": {
"function_tags": ["final_answer_emission"],
"depends_on": ["9, "30", "32"]
},
}
Here is the math problem:
<PROBLEM>
Here is the full chain-of-thought, broken into sentences:
<SENTENCES>
Now label each sentence with function tags and dependencies."""