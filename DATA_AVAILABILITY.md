# Data availability

| Dataset | Availability |
| --- | --- |
| CIC-IDS2017 (label-corrected, Engelen et al. 2021) | Public, from the authors' release. Not redistributed here. |
| UNSW-NB15 (official sets and PCAP) | Public, from UNSW Canberra. Not redistributed here. |
| PTMA campus flows | Pseudonymised flow records (keyed HMAC-SHA256 on IP addresses, no payloads) are available on reasonable request, subject to approval by the institution. Raw packet captures and original addresses are not shared. |

All scripts needed to rebuild the flow files from the public PCAPs are in `scripts/` and `tools/`.
The HMAC key is not part of this repository and must never be committed.
