# Tesco conversation data audit

This report is generated from code. It profiles the reconstructed public reply
graphs before taxonomy design or model training.

## Provenance

- Input: `tesco_messages.csv`
- Input SHA-256: `9663c4d57b506e7c9ec1ed443bb3685f8a394eaee9d86f0005230f3509f1f719`
- Source rows: 2811774
- Source SHA-256: `cd297fcfa1bf6f99938be242e8e578980bc6d1b96adc8691abec9a39175b03c0`

## Structure

| Measure | Value |
|---|---:|
| Messages | 73,159 |
| Conversations | 16,722 |
| Tesco messages | 38,573 |
| Customer/other messages | 34,586 |
| Median messages per conversation | 4.0 |
| 90th percentile messages per conversation | 7.0 |
| Maximum messages in one conversation | 551 |
| Maximum graph depth | 118 |
| Conversations with at least 3 messages | 69.54% |
| Rows whose parent is unavailable | 67 |

## Brand reply behavior

These are regex-based descriptive indicators, not quality labels.

| Indicator | Count | Share of Tesco messages |
|---|---:|---:|
| Direct replies to customer/other messages | 38,470 | 99.73% |
| Private-channel handoff language | 10,361 | 26.86% |
| Contains a URL | 3,190 | 8.27% |
| Numbered fragment such as 1/3 | 20,800 | 53.92% |
| Contains action language | 4,555 | 11.81% |
| Contains apology/empathy language | 11,403 | 29.56% |
| Contains a question mark | 15,560 | 40.34% |

Normalized unique reply share is 94.08%.
The ten most common normalized templates account for
1.22% of Tesco messages.

## Customer follow-up proxies

Only customer/other messages directly replying to a Tesco-authored message are
included here.

| Indicator | Count | Share of direct customer follow-ups |
|---|---:|---:|
| Direct follow-ups | 12,814 | 100.00% |
| Contains a positive cue | 3,373 | 26.32% |
| Contains an unresolved cue | 936 | 7.30% |
| Contains both | 178 | 1.39% |

Positive language is **not** a resolution label: “thanks, but it still does not
work” contains both signals, and “thanks, I sent a DM” confirms only a handoff.
These proxies are used to draw annotation candidates, never as golden truth.

## Date distribution

| Month | Messages |
|---|---:|
| 2014-09 | 1 |
| 2014-10 | 2 |
| 2015-01 | 3 |
| 2015-03 | 5 |
| 2015-07 | 4 |
| 2015-08 | 7 |
| 2015-11 | 6 |
| 2016-05 | 7 |
| 2016-06 | 4 |
| 2016-07 | 2 |
| 2016-08 | 11 |
| 2016-09 | 2 |
| 2016-10 | 4 |
| 2016-11 | 2 |
| 2016-12 | 3 |
| 2017-01 | 7 |
| 2017-02 | 5 |
| 2017-03 | 3 |
| 2017-04 | 1 |
| 2017-05 | 27 |
| 2017-06 | 20 |
| 2017-07 | 13 |
| 2017-08 | 28 |
| 2017-09 | 153 |
| 2017-10 | 30,636 |
| 2017-11 | 38,822 |
| 2017-12 | 3,381 |

## Consequences for the system

1. Reconstruct and split complete conversations before creating examples.
2. Join numbered brand fragments before treating them as evidence.
3. Separate public triage, private handoff, unknown outcome, explicit failure,
   and human-verified success.
4. Treat URLs and operational policy as historical artifacts, not current truth.
5. Require human review for food safety, allergens, injuries, refunds, private
   account access, and image-dependent complaints.
