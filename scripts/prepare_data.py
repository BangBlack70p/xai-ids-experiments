"""Prepare the UNSW-NB15 and PTMA flow datasets on the laptop (Windows, macOS or Linux).

Steps (run from the project folder, inside the .venv):

    python scripts/prepare_data.py check            # which tools and inputs are present
    python scripts/prepare_data.py build            # build the patched CICFlowMeter image (Docker)
    python scripts/prepare_data.py unsw             # UNSW-NB15 day-2 PCAPs -> flows -> GT labels
    python scripts/prepare_data.py ptma             # PTMA pcapng -> flows -> Suricata labels -> pseudonyms

Add --dry-run to print the commands without running them.

External tools
--------------
* Docker Desktop (runs CICFlowMeter; nothing Java-related is installed on Windows)
* Wireshark's editcap (converts .pcapng to .pcap; CICFlowMeter only accepts .pcap)
* Suricata for Windows + Npcap, and the ET Open rules archive (emerging.rules.tar.gz)

Default locations follow the project layout; override them with the options
shown by ``python scripts/prepare_data.py <step> --help``.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = ROOT / "data"
CFM_COMMIT = "4dd5319ad36457010d7a406505790b17a5828108"   # GintsEngelen/CICFlowMeter, June 2024
CFM_ZIP = f"https://github.com/GintsEngelen/CICFlowMeter/archive/{CFM_COMMIT}.zip"
IMAGE = "cicflowmeter-utc"
PY = sys.executable

WIN_CANDIDATES = {
    "editcap": [r"C:\Program Files\Wireshark\editcap.exe"],
    "suricata": [r"C:\Program Files\Suricata\suricata.exe"],
    "docker": [r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"],
}


def find_tool(name: str) -> str | None:
    hit = shutil.which(name)
    if hit:
        return hit
    for cand in WIN_CANDIDATES.get(name, []):
        if Path(cand).exists():
            return cand
    return None


LOG = ROOT / "logs" / "prepare_data.log"


def log(text: str) -> None:
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(text + "\n")


def run(cmd: list[str], dry: bool) -> None:
    """Run a command, echoing its output to the console and to logs/prepare_data.log."""
    log("  $ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    if dry:
        return
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        log("    " + line.rstrip())
    if proc.wait() != 0:
        log(f"ERROR: command failed with exit code {proc.returncode}. Full output: {LOG}")
        sys.exit(proc.returncode)


DRY = False


def need(name: str) -> str:
    tool = find_tool(name)
    if not tool and DRY:
        return f"<{name}>"
    if not tool:
        sys.exit(f"ERROR: {name} not found. See README section 'Menyiapkan data'.")
    return tool


# ------------------------------------------------------------------ steps
def step_check(_a) -> None:
    print("Tools")
    for t in ("docker", "editcap", "suricata"):
        log(f"  {t:9s} {find_tool(t) or 'NOT FOUND'}")
    print("Inputs")
    for p in [ROOT / "raw" / "dataset.zip", ROOT / "raw" / "NUSW-NB15_GT.csv",
              ROOT / "raw" / "pcaps 17-2-2015", ROOT / "raw" / "PCAP.zip",
              ROOT / "raw" / "Scrtadst" / "emerging.rules.tar.gz",
              ROOT / "raw" / "Training and Testing Sets UNSW-NB15"]:
        log(f"  {'ok ' if p.exists() else 'MISSING'} {p.relative_to(ROOT)}")
    docker = find_tool("docker")
    if docker:
        r = subprocess.run([docker, "image", "inspect", IMAGE], capture_output=True)
        log(f"  image {IMAGE}: {'built' if r.returncode == 0 else 'not built yet (run: build)'}")
        r = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True)
        log(f"  docker engine: {'running, version ' + r.stdout.strip() if r.returncode == 0 else 'NOT RUNNING - start Docker Desktop'}")


def step_build(a) -> None:
    docker = need("docker")
    src = ROOT / "tools" / "CICFlowMeter-src"
    git = find_tool("git")
    if not (src / "build.gradle").exists() and git:
        src.parent.mkdir(parents=True, exist_ok=True)
        run([git, "-c", "core.autocrlf=false", "clone", "https://github.com/GintsEngelen/CICFlowMeter", src],
            a.dry_run)
        run([git, "-C", src, "checkout", "-q", CFM_COMMIT], a.dry_run)
    elif not (src / "build.gradle").exists():
        print(f"downloading {CFM_ZIP}")
        if not a.dry_run:
            src.parent.mkdir(parents=True, exist_ok=True)
            data = urllib.request.urlopen(CFM_ZIP, timeout=120).read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                top = zf.namelist()[0].split("/")[0]
                zf.extractall(src.parent)
            if src.exists():
                shutil.rmtree(src)
            (src.parent / top).rename(src)
    run([docker, "build", "-f", ROOT / "tools" / "cicflowmeter" / "Dockerfile", "-t", IMAGE, src], a.dry_run)


def _cfm(docker: str, folder: Path, out: Path, mem_gb: int, dry: bool) -> None:
    run([docker, "run", "--rm", "-e", f"JAVA_OPTS=-Xmx{mem_gb}g",
         "-v", f"{folder.resolve()}:/in:ro", "-v", f"{out.resolve()}:/out", IMAGE, "/in", "/out"], dry)


def extract_flows(pcaps: list[Path], out: Path, mem_gb: int, dry: bool) -> None:
    """Run CICFlowMeter on every pcap.

    * .pcapng is converted to .pcap with editcap (CICFlowMeter only accepts .pcap);
    * Linux "cooked" captures (link type 113, e.g. UNSW-NB15) are rewritten to
      Ethernet one file at a time, processed, and the converted copy deleted, so
      the extra disk space never exceeds one file.
    """
    from sll_to_ether import LINKTYPE_LINUX_SLL, convert, linktype
    docker = need("docker")
    stage = DATA / "_pcap_stage"
    out.mkdir(parents=True, exist_ok=True)
    folders: dict[Path, list[Path]] = {}
    for p in pcaps:
        if p.suffix.lower() == ".pcapng":
            editcap = need("editcap")
            stage.mkdir(parents=True, exist_ok=True)
            target = stage / (p.stem + ".pcap")
            if not target.exists():
                run([editcap, "-F", "pcap", p, target], dry)
            folders.setdefault(stage, []).append(target)
        elif not dry and linktype(p) == LINKTYPE_LINUX_SLL:
            one = stage / "sll" / p.stem
            one.mkdir(parents=True, exist_ok=True)
            target = one / p.name
            log(f"  converting Linux SLL -> Ethernet: {p.name}")
            st = convert(p, target, progress_every=0)
            log(f"    {st['converted']:,} of {st['packets']:,} packets converted")
            try:
                _cfm(docker, one, out, mem_gb, dry)
            finally:
                target.unlink(missing_ok=True)
        else:
            folders.setdefault(p.parent, []).append(p)
    for folder in folders:
        _cfm(docker, folder, out, mem_gb, dry)


def suricata(pcap: Path, out: Path, rules_tgz: Path, dry: bool, home_net: str | None = None) -> Path:
    exe = need("suricata")
    out.mkdir(parents=True, exist_ok=True)
    rules = out / "et_open.rules"
    if not dry and not rules.exists():
        with tarfile.open(rules_tgz) as tf, open(rules, "w", encoding="utf-8") as fh:
            n = 0
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".rules"):
                    fh.write(tf.extractfile(m).read().decode("utf-8", "replace") + "\n")
                    n += 1
        print(f"  combined {n} rule files into {rules}")
    yaml = Path(exe).with_name("suricata.yaml")
    cmd = [exe, "-r", pcap, "-l", out, "-S", rules, "-k", "none"]
    if home_net:
        # Inbound ET rules are written as $EXTERNAL_NET -> $HOME_NET; a campus server on a
        # public prefix must be in HOME_NET or attacks against it never match.
        cmd += ["--set", f"vars.address-groups.HOME_NET={home_net}"]
    eve = out / "eve.json"
    if eve.exists() and not dry:
        eve.unlink()
    if yaml.exists():
        cmd[1:1] = ["-c", yaml]
    run(cmd, dry)
    return out / "eve.json"


def step_unsw(a) -> None:
    pcaps = sorted(Path(a.pcaps).glob("*.pcap"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    print(f"UNSW-NB15: {len(pcaps)} pcaps in {a.pcaps}")
    raw, lab = DATA / "unsw_flows_raw", DATA / "unsw_flows_labelled"
    if not a.skip_extract:
        extract_flows(pcaps, raw, a.mem_gb, a.dry_run)
    run([PY, ROOT / "scripts" / "label_flows.py", "--flows", raw, "--schedule", a.gt,
         "--gt-unsw", "--bidirectional", "--tolerance", "2", "--out", lab], a.dry_run)
    run([PY, ROOT / "scripts" / "check_flows.py", lab], a.dry_run)


def step_ptma(a) -> None:
    pcap = Path(a.pcap)
    if pcap.suffix.lower() == ".zip":
        unz = DATA / "ptma_pcap"
        print(f"unzipping {pcap} -> {unz}")
        if not a.dry_run:
            with zipfile.ZipFile(pcap) as zf:
                zf.extractall(unz)
        found = sorted(unz.rglob("*.pcap*")) if not a.dry_run else [unz / "PCAP" / "today.pcapng"]
        pcap = found[0]
    raw, sur = DATA / "ptma_flows_raw", DATA / "suricata_ptma"
    lab_raw, lab = DATA / "ptma_flows_labelled_raw", DATA / "ptma_flows_labelled"
    if not a.skip_extract:
        extract_flows([pcap], raw, a.mem_gb, a.dry_run)
    eve = suricata(pcap, sur, Path(a.rules), a.dry_run, a.home_net) if not a.skip_suricata else sur / "eve.json"
    schedule = DATA / "ptma_suricata_schedule.csv"
    run([PY, ROOT / "scripts" / "suricata_to_schedule.py", "--eve", eve, "--out", schedule,
         "--max-severity", str(a.max_severity), "--report", DATA / "ptma_suricata_signatures.csv"]
        + (["--exclude-src", *a.exclude_src] if a.exclude_src else []), a.dry_run)
    run([PY, ROOT / "scripts" / "label_flows.py", "--flows", raw, "--schedule", schedule,
         "--bidirectional", "--tolerance", "2", "--out", lab_raw], a.dry_run)
    run([PY, ROOT / "scripts" / "pseudonymize.py", "--in", lab_raw, "--out", lab,
         "--key-file", a.key_file, "--internal", *a.internal], a.dry_run)
    run([PY, ROOT / "scripts" / "check_flows.py", lab], a.dry_run)
    print(f"\nNOTE: {lab_raw} still holds real IP addresses. Delete it once {lab} is checked.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name in ("check", "build", "unsw", "ptma"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        if name in ("unsw", "ptma"):
            p.add_argument("--mem-gb", type=int, default=4, help="Java heap for CICFlowMeter")
            p.add_argument("--skip-extract", action="store_true", help="reuse existing raw flows")
        if name == "unsw":
            p.add_argument("--pcaps", default=str(ROOT / "raw" / "pcaps 17-2-2015"))
            p.add_argument("--gt", default=str(ROOT / "raw" / "NUSW-NB15_GT.csv"))
        if name == "ptma":
            p.add_argument("--pcap", default=str(ROOT / "raw" / "PCAP.zip"))
            p.add_argument("--rules", default=str(ROOT / "raw" / "Scrtadst" / "emerging.rules.tar.gz"))
            p.add_argument("--max-severity", type=int, default=2)
            p.add_argument("--skip-suricata", action="store_true")
            p.add_argument("--key-file", default=str(Path.home() / ".xai_ids" / "ptma_hmac.key"))
            p.add_argument("--exclude-src", nargs="*", default=[],
                           help="source IPs confirmed legitimate by PTMA IT (never labelled as attack)")
            p.add_argument("--home-net", default="[10.0.0.0/8,172.16.0.0/12,192.168.0.0/16]",
                           help="Suricata HOME_NET; include PTMA's public prefix")
            p.add_argument("--internal", nargs="*", default=[],
                           help="public prefixes that belong to PTMA (see audit report)")
    a = ap.parse_args()
    import datetime
    log(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  prepare_data.py {' '.join(sys.argv[1:])} =====")
    global DRY
    DRY = a.dry_run
    os.chdir(ROOT)
    {"check": step_check, "build": step_build, "unsw": step_unsw, "ptma": step_ptma}[a.step](a)


if __name__ == "__main__":
    main()
