# Optional: SentiStrength setup

TrendScope AI can use **SentiStrength** (M. Thelwall et al., University of Wolverhampton, http://sentistrength.wlv.ac.uk/) as one of its sentiment algorithms (`sentistrength`).

SentiStrength is a separate third-party product with its own licence terms. Its jar, manual and language data are **not included in this repository** and are not covered by this repository's MIT licence. The integration code (`sentiStrength.py`, `trendscope/sentiment.py`) is included. Without the jar, TrendScope still works: the `sentistrength` option is reported as unavailable and the `vader` and `lexicon` algorithms are unaffected.

## Setup

1. Read the SentiStrength licence on the site above and obtain the Java version and the English data from the authors.
2. Install Java and make sure `java` is on your `PATH`.
3. Place the files in the project root with exactly this layout:

```
SentiStrength/
  SentiStrengthCom.jar
  SentiStrength_DataEnglishFeb2017/     (the English data folder)
```

4. Restart TrendScope. The `sentistrength` algorithm should now show as available.

The `SentiStrength/` folder is listed in `.gitignore`, so it is never committed by accident.
