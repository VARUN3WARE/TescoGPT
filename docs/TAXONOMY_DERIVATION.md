# Tesco intent-taxonomy derivation audit

> Generated from training-period cases only. This is an overlapping lexical 
> coverage audit, not unsupervised discovery and not human ground truth.

## Provenance

- Input: `tesco_cases.csv`
- Input SHA-256: `dce80d676b7a1979b5a8454b24ef86d5be12c2fbbad26a2ef497b627e65481bb`
- Deterministic example seed: `20260913`

## Why these intent boundaries

The taxonomy uses operational support actions rather than product categories. Closely related surface forms are consolidated when they need the same queue or public response: pricing, promotions and Clubcard form one family; login, app and checkout failures form another. Product defects and safety reports share one intent, while risk tags and escalation reasons preserve the safety distinction. Standalone refund process questions remain separate, but a refund caused by delivery or a defect is secondary to that underlying issue. `other_or_unclear` is the residual, not a dictionary theme.

## Training-data coverage

Training cases: **17,345**. Cases matching no theme: **8,597**. Cases matching multiple themes: **2,276**.

| Candidate intent family | Lexical matches | Share of training cases |
|---|---:|---:|
| `delivery_or_collection` | 1,320 | 7.61% |
| `product_quality_or_safety` | 408 | 2.35% |
| `product_availability` | 1,023 | 5.90% |
| `pricing_promotion_or_clubcard` | 1,058 | 6.10% |
| `refund_return_or_exchange` | 501 | 2.89% |
| `online_account_or_checkout` | 1,458 | 8.41% |
| `store_or_staff_experience` | 2,658 | 15.32% |
| `product_information` | 509 | 2.93% |
| `feedback_praise_or_suggestion` | 2,536 | 14.62% |

Counts overlap. High overlap is evidence for explicit tie-break rules, not a reason to report these matches as classifier performance.

## Deterministic examples

### `delivery_or_collection`

- `tesco-2508669` — Thought I’d try @Tesco home delivery for first time rather than my @customer to see if more cost effective. big mistake. Wasn’t any cheaper once I paid the “service charge” order was 15mins late, no heads up until driver phoned as was lost. #dissapointingservice
- `tesco-1109002` — When you can’t sleep so you do your @Tesco shop at 4am for delivery the same day.... #winning 👌🏻 #itsthesmallthings #reallyneedtosleepnow
- `tesco-2403061` — @Tesco so when you claim its next day delivery that's in fact a big lie!!!!

### `product_quality_or_safety`

- `tesco-1979160` — Can someone tell me why the jelly I just bought from Tesco has mould inside?! @Tesco <media_or_link>
- `tesco-1024094` — @Tesco ill you sell the Apple watch series 3 anytime soon?
- `tesco-1476542` — @Tesco Hi <name_redacted>, sadly I do not live in Loughborough and was travelling down to London. I have times and photos of my injured knee.

### `product_availability`

- `tesco-2145466` — @Tesco Hi there, I was just wondering if I could check if COD WW2 is in stock in your Hexham store, I went in a few days ago and it wasn't, just wondering if that's still the case?
- `tesco-1722189` — Really struggling with online shop since update, @Tesco. Tons of products no longer available so I have to go to the shop. Kinda pointless!
- `tesco-1041469` — @Tesco your Uxbridge metro store no longer sell Tesco's own salt + vinegar chipsticks, do you know if they'll be back?? :(

### `pricing_promotion_or_clubcard`

- `tesco-2508669` — Thought I’d try @Tesco home delivery for first time rather than my @customer to see if more cost effective. big mistake. Wasn’t any cheaper once I paid the “service charge” order was 15mins late, no heads up until driver phoned as was lost. #dissapointingservice
- `tesco-1619662` — @Tesco @Tesco... seen as I can't make it in to collect what was offered... could you send time or a gift card
- `tesco-862023` — @sainsburys I’m guessing you’ll offer me a much better servce than @Tesco?! After 7 years of loyal shopping I am DONE! #getmeanectarcard

### `refund_return_or_exchange`

- `tesco-987887` — @Tesco Hi guys, I've found a 'Colleague Room Payment Card' on the pavement outside. Please could you advise where it should be returned to?
- `tesco-264619` — @Tesco - 3-5 days for a refund that I was told wouldn't come out of my account in the first place. 8 days later, still no refund. #Shocking
- `tesco-751556` — Why have all of the Brazil Nuts in @Tesco been replaced with Pecans this year??? 🤔 Not a single Brazil Nut in sight! <media_or_link>

### `online_account_or_checkout`

- `tesco-987887` — @Tesco Hi guys, I've found a 'Colleague Room Payment Card' on the pavement outside. Please could you advise where it should be returned to?
- `tesco-1722189` — Really struggling with online shop since update, @Tesco. Tons of products no longer available so I have to go to the shop. Kinda pointless!
- `tesco-264619` — @Tesco - 3-5 days for a refund that I was told wouldn't come out of my account in the first place. 8 days later, still no refund. #Shocking

### `store_or_staff_experience`

- `tesco-987887` — @Tesco Hi guys, I've found a 'Colleague Room Payment Card' on the pavement outside. Please could you advise where it should be returned to?
- `tesco-740855` — Class 3 had a fantastic time at @Tesco yesterday on the farm to fork trail. They looked at and tried a variety of different foods across the store including the bakery and fish department! 🐟 🍞 🥒 🍖 <media_or_link>
- `tesco-2145466` — @Tesco Hi there, I was just wondering if I could check if COD WW2 is in stock in your Hexham store, I went in a few days ago and it wasn't, just wondering if that's still the case?

### `product_information`

- `tesco-2466526` — Thank you @Tesco for making this ice cream! #wheatfree #glutenfree #vegetarian <media_or_link> … <media_or_link>
- `tesco-2473831` — How shocked was I when I popped into my decent size @Tesco last night to buy a tin opener and - nothing? Isn't that a basic thing they would stock?!!
- `tesco-2291833` — @tesco - think your butcher counter staff need more practice. 2x400g steaks ordered do not equal 580g total weight delivered 😞

### `feedback_praise_or_suggestion`

- `tesco-2143200` — @customer @Tesco Before we, as a family, decide where to buy our Christmas Turkey, can you please tell me if your turkeys are Halal? Thank you.
- `tesco-1818642` — @Tesco Thanks for clarifying 🤔
- `tesco-2466526` — Thank you @Tesco for making this ice cream! #wheatfree #glutenfree #vegetarian <media_or_link> … <media_or_link>

## What this does not prove

These dictionaries were written to audit chosen operational families; they do not claim the taxonomy emerged automatically. Polysemy, negation, multi-intent cases and polite words create false matches. The blind 200-case human annotation is the only source of evaluation labels, and its disagreement analysis is the test of whether these boundaries are usable.
