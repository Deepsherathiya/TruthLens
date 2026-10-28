"""Method: how both signals work, what is sent online, and the limits."""

from core import evidence
from core.style_model import UNCERTAIN_BAND
from ui import components as ui, state


def render() -> None:
    lo, hi = UNCERTAIN_BAND
    key_status = ("is set" if state.factcheck_key() else
                  "is not set, so published fact-checks are not searched")
    ui.page_header(
        "Method", "How TruthLens reads a headline",
        "Two independent signals, each simple enough to explain step by step. Neither is a fact-check "
        "on its own; together they tell you how a headline is written and whether anyone else reports it.",
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
    <li><strong>Classifier.</strong> Logistic regression, trained on 80% of the ISOT dataset after
      removing duplicate headlines and headlines that appear in both classes. Multinomial Naive Bayes
      is trained on the same split for comparison.</li>
    <li><strong>Uncertain zone.</strong> If P(fake) is between {lo:.0%} and {hi:.0%}, or none of the
      words were seen in training, the answer is <em>Uncertain</em> instead of a forced call.</li>
    <li><strong>Explanation.</strong> Each word's contribution is its TF-IDF value times the model's
      coefficient; positive pushes toward fake.</li>
  </ol>

  <h2>2 · The source check</h2>
  <ol>
    <li><strong>Key terms.</strong> Stop-words and reporting words (“says”, “breaking”) are dropped.
      Numbers weigh {evidence.WEIGHT_NUMBER:g}, names and acronyms {evidence.WEIGHT_NAME:g},
      other words {evidence.WEIGHT_WORD:g}. A light suffix rule makes “passes” match “passed”.</li>
    <li><strong>Search.</strong> The key terms are sent to Wikipedia (article intros), Wikipedia's
      daily <em>Current events</em> archive (each item cites the outlet that reported it, such as
      Reuters or the BBC), GDELT (worldwide news from about the last three months) and, if an API key
      is configured, Google Fact Check Tools.</li>
    <li><strong>Matching.</strong> A passage is a <em>close match</em> when it contains at least
      {evidence.STRONG_MATCH:.0%} of the weighted key terms and most of the names and numbers.
      If a negation (“not”, “never”, “denies”) comes just before a matched word in only one of the
      two texts, it is marked <em>opposite wording</em>. Weaker overlaps are shown as <em>related</em> but not counted.</li>
    <li><strong>Verdict.</strong> Fact-checkers' ratings of a matching claim decide first. Otherwise
      <em>Corroborated</em> needs close matches from at least {evidence.MIN_INDEPENDENT} independent
      publishers (or one Wikipedia passage covering {evidence.WIKI_ALONE:.0%} of the terms), and
      <em>Disputed</em> needs {evidence.MIN_INDEPENDENT} publishers with opposite wording. Anything
      else is <em>Not enough evidence</em>.</li>
  </ol>

  <h2>What leaves your computer</h2>
  <p>The writing-style model runs entirely on this machine. When <strong>Cross-check online
  sources</strong> is on, the headline's key terms (not your history) are sent to the services
  above. Results are cached for an hour so repeated checks don't call them again. The Google Fact
  Check key {key_status}; add it as <code>GOOGLE_FACTCHECK_API_KEY</code> in the environment or in
  <code>.streamlit/secrets.toml</code>.</p>

  <h2>Limits worth knowing</h2>
  <ul>
    <li><strong>Style is not truth.</strong> The model learned how 2016–17 Reuters headlines and
      fake-news sites write. A sober fake or a sensational real story will fool it.</li>
    <li><strong>Old, narrow training data.</strong> Mostly US politics. New names are unknown words,
      and the app warns when few words are recognised.</li>
    <li><strong>Word matching, not understanding.</strong> The source check can miss paraphrases
      (“approves” vs “passes”) and contradictions that don't use a negation word. A missing match
      is not a debunk, especially for very recent or local news.</li>
    <li><strong>Free APIs.</strong> GDELT allows one request every five seconds and is often busy;
      the check simply reports which sources answered.</li>
  </ul>
</article>""")
    ui.footer()
