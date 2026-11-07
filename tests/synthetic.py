"""Synthetic screenplays for tests (the real ones are confidential and never committed)."""
from pathlib import Path


def synthetic_episode(n: int) -> str:
    """60s episode: an action line 0-4s, then a dialogue line every 4 s (4.00-7.50, 8.00-11.50 ...)."""
    out = [f"EPISODE NUMBER: Episode {n}", f"EPISODE LOGLINE: Synthetic episode {n}.",
           "--- SCRIPT START ---", "SCENE 01", "Scene heading: INT. HALL", "Scene start: 00:00.00",
           "Scene end: 01:00.00", "", "**ACTION:**", "Start: (00:00.00) | End: (00:04.00)", "A door slams.", ""]
    for i in range(1, 14):
        s, e = i * 4, i * 4 + 3.5
        who = "MARA (Angry)" if i % 2 else "TOMAS (Cold)"
        out += [f"**{who}:**", f"Start: (00:{s:02d}.00) | End: (00:{int(e):02d}.50)", f'"Line {i} of episode {n}."', ""]
    out.append("--- SCRIPT END ---")
    return "\n".join(out)


def write_show(folder: Path, episodes=(1, 2, 3)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for n in episodes:
        (folder / f"episode-{n:03d}.txt").write_text(synthetic_episode(n), encoding="utf-8")
    return folder
