"""Convert a Linux "cooked" capture (pcap linktype 113, SLL) to Ethernet (linktype 1).

UNSW-NB15's PCAPs were captured on the Linux "any" interface, so every packet
starts with a 16-byte SLL header instead of a 14-byte Ethernet header.
CICFlowMeter (jnetpcap) only decodes Ethernet and silently discards every SLL
packet. This script rewrites each packet header:

    SLL:      pkttype(2) hatype(2) halen(2) addr(8) protocol(2)
    Ethernet: dst(6)=00:00:00:00:00:00  src(6)=first 6 bytes of SLL addr  type(2)=protocol

Payloads, timestamps and packet order are untouched, so flows and their
timing are identical. Streams the file; memory use stays small.

    python scripts/sll_to_ether.py "raw/pcaps 17-2-2015/1.pcap" data/_pcap_stage/1.pcap
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

LINKTYPE_ETHERNET = 1
LINKTYPE_LINUX_SLL = 113
MAGICS = {b"\xd4\xc3\xb2\xa1": "<", b"\xa1\xb2\xc3\xd4": ">",       # microsecond
          b"\x4d\x3c\xb2\xa1": "<", b"\xa1\xb2\x3c\x4d": ">"}       # nanosecond


def linktype(path: str | Path) -> int | None:
    """Link type of a classic pcap file, or None for pcapng / unknown files."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    endian = MAGICS.get(head[:4])
    if endian is None or len(head) < 24:
        return None
    return struct.unpack(endian + "I", head[20:24])[0]


def convert(src: str | Path, dst: str | Path, progress_every: int = 1_000_000) -> dict:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    stats = {"packets": 0, "converted": 0, "short": 0}
    with open(src, "rb", buffering=8 << 20) as fi, open(dst, "wb", buffering=8 << 20) as fo:
        head = fi.read(24)
        endian = MAGICS.get(head[:4])
        if endian is None:
            raise ValueError(f"{src}: not a classic pcap file")
        if struct.unpack(endian + "I", head[20:24])[0] != LINKTYPE_LINUX_SLL:
            raise ValueError(f"{src}: link type is not Linux SLL (113)")
        rec = struct.Struct(endian + "IIII")
        fo.write(head[:20] + struct.pack(endian + "I", LINKTYPE_ETHERNET))
        zero_mac = b"\x00" * 6
        while True:
            hdr = fi.read(16)
            if len(hdr) < 16:
                break
            ts_s, ts_frac, incl, orig = rec.unpack(hdr)
            data = fi.read(incl)
            if len(data) < incl:          # truncated last record
                break
            stats["packets"] += 1
            if incl < 16:
                stats["short"] += 1
                continue
            src_mac = data[6:12]          # SLL address field (first 6 of 8 bytes)
            eth = zero_mac + src_mac + data[14:16] + data[16:]
            fo.write(rec.pack(ts_s, ts_frac, incl - 2, max(orig - 2, 0)) + eth)
            stats["converted"] += 1
            if progress_every and stats["packets"] % progress_every == 0:
                print(f"    {src.name}: {stats['packets']:,} packets", flush=True)
    return stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    s = convert(sys.argv[1], sys.argv[2])
    print(f"{sys.argv[1]} -> {sys.argv[2]}: {s['converted']:,} of {s['packets']:,} packets converted "
          f"({s['short']} too short)")
