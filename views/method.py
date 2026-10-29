"""Method: how both signals work, what is sent online, and the limits."""

from core import evidence, nli_evidence, nlp_models
from core.style_model import MIN_WORDS, UNCERTAIN_BAND
from ui import components as ui, state


def render() -> None:
    lo, hi = UNCERTAIN_BAND
    key_status = ("is set" if state.factcheck_key() else
                  "is not set, so published fact-checks are not searched")
    engine_status = ("installed, so the NLI engine is used" if nlp_models.available() else
                     "not installed, so the word-matching engine is used")
    ui.page_header(
        "Method", "How TruthLens reads a headline",
        "Two independent signals: how a headline is written, and what the evidence says about it. "
        "Neither is a guarantee on its own.",
    )
    ui.show(f"""
<article class="tl-prose">
  <h2>1 · The writing-style model</h2>
  <ol>
    <li><strong>Cleaning.</strong> The headline is lowercased and stripped of tags that reveal which
      website it came from (<code>[VIDEO]</code>, <code>BREAKING:</code>, <code>Factbox:</code>) and
      <code>U.S.</code> becomes <code>us</code>. Negations such as “not” are kept. Training and
      prediction use the same function.</li>
    <li><strong>Features.</strong> TF-IDF over single words and two-word phrases seen in at least two
      training headlines.</li>
    <li><strong>Classifier.</strong> Logistic regression, trained on 80% of two datasets after
      removing duplicate headlines and headlines that appear in both classes: ISOT (US news, 2016–17)
      and IFND (Indian news, 2013–21). Most of IFND's fake headlines are fact-checkers' titles
      (“Fact Check: No, this photo is not…”), so words such as “fact check”, “false” and “viral” are
      removed from it first; otherwise the model would learn to spot fact-check articles. Multinomial
      Naive Bayes is trained on the same split for comparison. Each model's main setting (regularisation
      strength C, smoothing alpha) is chosen on a validation slice of the training data; the test set
      is used once, at the end.</li>
    <li><strong>Uncertain zone.</strong> If P(fake) is between {lo:.0%} and {hi:.0%}, none of the
      words were seen in training, or the text has fewer than {MIN_WORDS} words (“modi died” has no
      writing style to judge), the answer is <em>Uncertain</em> instead of a forced call.</li>
    <li><strong>Explanation.</strong> Each word's contribution is its TF-IDF value times the model's
      coefficient; positive pushes toward fake.</li>
  </ol>

  <h2>2 · The evidence check</h2>
  <ol>
    <li><strong>Search.</strong> Published fact-checks (Google Fact Check Tools, if an API key is set),
      the top Wikipedia articles, Wikipedia's daily <em>Current events</em> archive (each item cites the
      outlet that reported it, such as Reuters or the BBC) and GDELT (worldwide news from about the
      last three months). Simple yes/no questions are first rewritten as statements. For an article
      about a person, Wikipedia's categories add one sentence such as “Narendra Modi is alive.”
      (<em>Living people</em>) or “Ratan Tata died in 2024.” (<em>2024 deaths</em>), so death hoaxes
      can be refuted even though no article says “he did not die”. A death claim that gives only a
      surname (“modi died”) is checked about the best-known person with that name (the Wikipedia
      biography most other articles link to), and the app says who it checked.</li>
    <li><strong>Fact-checks.</strong> A fact-check only counts if it is about the <em>same</em> claim:
      similar meaning and, in the logically valid direction, one statement implies the other
      (rated False: our claim must imply the checked claim; rated True: the reverse).</li>
    <li><strong>Ranking.</strong> Sentence embeddings (MiniLM) keep the {nli_evidence.CANDIDATES}
      sentences most similar to the claim.</li>
    <li><strong>NLI.</strong> A DeBERTa-v3 cross-encoder reads each sentence with the claim and gives
      probabilities for <em>entails</em>, <em>contradicts</em> and <em>neutral</em>.</li>
    <li><strong>Verdict.</strong> Matching fact-checks decide first. Otherwise the <em>most relevant</em>
      sentence (similarity ≥ {nli_evidence.RELEVANCE}) that the model is confident about
      (≥ {nli_evidence.STANCE_PROB:.0%}) decides: <em>Supported</em> or <em>Refuted</em>. Going in
      relevance order avoids the model's habit of calling off-topic sentences contradictions.
      The deciding sentence must also contain at least {nli_evidence.KEY_TERMS:.0%} of the claim's
      key words and {nli_evidence.KEY_NAMES:.0%} of its names and numbers, so a sentence about
      someone else (“Minister Arvind Sawant resigned”) can't decide “Narendra Modi resigned”.
      A <em>refuting</em> sentence must also share a word besides the names, because contradiction is
      the NLI model's weak spot: “Modi is often accused of having narcissistic traits” mentions Modi
      but says nothing about “modi is a dictator”. Anything else is <em>Not enough evidence</em>.</li>
  </ol>

  <h3>Fallback: word matching</h3>
  <p>The NLI models need PyTorch (currently {engine_status}). Without it, each passage is judged by
  how many of the claim's key words it contains: numbers weigh {evidence.WEIGHT_NUMBER:g}, names
  {evidence.WEIGHT_NAME:g}, other words {evidence.WEIGHT_WORD:g}. A <em>close match</em> needs
  {evidence.STRONG_MATCH:.0%} of them and most names and numbers; a negation (“not”, “never”) just
  before a matched word in only one of the two texts makes it <em>opposite wording</em>.
  <em>Supported</em> then needs close matches from {evidence.MIN_INDEPENDENT} independent publishers,
  and <em>Refuted</em> needs {evidence.MIN_INDEPENDENT} with opposite wording.</p>

  <h2>What leaves your computer</h2>
  <p>The writing-style model and the NLI models run on this machine. When <strong>Check evidence
  online</strong> is on, the claim (or its key words) is sent to the services above. Results are
  cached for an hour so repeated checks don't call them again. The Google Fact Check key {key_status};
  put <code>GOOGLE_FACTCHECK_API_KEY=…</code> in <code>.env</code> (see <code>.env.example</code>).</p>

  <h2>Limits worth knowing</h2>
  <ul>
    <li><strong>Style is not truth.</strong> The model learned how Reuters and Indian newspapers
      write headlines, and how fake-news sites and fact-checked Indian claims are worded. A sober fake
      or a sensational real story will fool it.</li>
    <li><strong>Retrieval is the weak link.</strong> If the wrong article is found, the NLI step reasons
      correctly about the wrong text. The thresholds were tuned on 10 demo claims, so real accuracy is
      lower.</li>
    <li><strong>Numbers and dates</strong> are hard for small NLI models.</li>
    <li><strong>Recent or local news</strong> may not be covered yet. <em>Not enough evidence</em> is
      not a debunk.</li>
    <li><strong>Free APIs.</strong> GDELT allows one request every five seconds and is often busy;
      the check reports which sources answered.</li>
  </ul>
</article>""")
    ui.footer()
