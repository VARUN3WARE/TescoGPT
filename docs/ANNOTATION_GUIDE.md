# TescoGPT annotation guide

**Codebook version:** 1.0
**Unit:** one incoming public customer message, plus only the conversation context
that existed before that message.

The annotation sheets deliberately hide Tesco's later historical response,
whether a case belongs to the natural or challenge slice, and the heuristics
used to select challenge cases. Do not search for the original tweet or infer
the label from how Tesco happened to respond. Label what the proposed agent
should do from the information available at decision time.

## Required fields

| Field | Values |
|---|---|
| `intent_label` | Exactly one primary intent from the taxonomy below |
| `secondary_intent` | Optional second intent; blank if none |
| `handling_label` | `AUTO_HANDLE` or `ESCALATE` |
| `reason_code` | Exactly one reason compatible with the handling label |
| `risk_tags` | Zero or more semicolon-separated risk tags |
| `must_include` | Essential content or action for an acceptable public reply |
| `must_avoid` | Claims, requests, or actions an acceptable reply must avoid |
| `annotator_id` | Stable non-empty identifier, not a real full name |
| `annotation_notes` | Brief explanation for ambiguity or unusual decisions |

## Intent taxonomy

### `delivery_or_collection`

Online grocery delivery, Click & Collect, delivery slots, drivers, substitutions,
missing bags/items, late or cancelled delivery, or an order arriving at the wrong
place.

- “Two bags from my order did not arrive.”
- “Why was my delivery slot cancelled?”
- A missing item in an online order remains this intent even if a refund is also
  requested; use `refund_return_or_exchange` as secondary.

### `product_quality_or_safety`

Spoilage, damage, contamination, foreign objects, tampering, incorrect contents,
poor freshness, illness, injury, choking, allergic reaction, or another concern
about whether a product is safe or fit for use.

- “There was glass in this jar.”
- “The chicken was raw and made us ill.”
- Product quality is primary when the defect caused a refund request.

### `product_availability`

Whether Tesco stocks a product, a discontinued range, local availability, or
when an item might return.

- “Do you still sell the prosecco crisps?”
- A product missing from a completed online delivery is
  `delivery_or_collection`, not availability.

### `pricing_promotion_or_clubcard`

Shelf/checkout price differences, promotions, coupons, Clubcard prices or points,
gift cards, vouchers, and discount eligibility.

- “The Clubcard price was not applied.”
- An already-issued refund or return is not this intent merely because money is
  mentioned.

### `refund_return_or_exchange`

The process, status, eligibility, or execution of a refund, return, replacement,
or exchange when no more specific underlying operational issue is primary.

- “How can I return an unopened electrical item?”
- “Where is the refund you already approved?”

### `online_account_or_checkout`

Login, password, account access, website/app failure, payment failure, basket,
checkout, or an online order that cannot be placed.

- “The app signs me out every time I check out.”
- A successfully placed order that later goes missing is
  `delivery_or_collection`.

### `store_or_staff_experience`

Staff conduct, queues, cleanliness, facilities, opening hours, accessibility,
trolleys, parking, or a complaint/praise about a physical store visit.

- “There were no accessible trolleys at the entrance.”
- Product stock remains `product_availability` unless the main complaint is how
  store staff handled it.

### `product_information`

Ingredients, nutrition, allergens without a reported reaction, packaging,
preparation, compatibility, product differences, sourcing, or other factual
questions about an item.

- “Is this soup suitable for vegans?”
- A reported allergic reaction is `product_quality_or_safety`.

### `feedback_praise_or_suggestion`

Thanks, praise, general product feedback, campaign discussion, or a suggestion
that does not ask for individual support action.

- “Your driver was brilliant today.”
- “Please bring back the old recipe.”

### `other_or_unclear`

Spam, unrelated conversation, unintelligible text, unsupported language, or a
message whose operational intent cannot be determined from the available
context.

Do not use this label simply because a normal case is difficult.

## Primary-intent tie-break order

When several labels apply, choose the operational issue that caused the request:

1. product safety/quality;
2. delivery/collection;
3. online account/checkout;
4. store/staff experience;
5. pricing/promotion/Clubcard;
6. refund/return/exchange;
7. product availability;
8. product information;
9. feedback/praise/suggestion;
10. other/unclear.

Record a genuine second issue in `secondary_intent`. This field is diagnostic;
the headline intent metric uses only the primary label.

## Handling policy

`AUTO_HANDLE` means the public draft may be sent without pre-send review. It does
not mean the underlying case is closed. Choose it only when a safe public reply
can be written from the visible information without private data, a backend
lookup, a discretionary promise, image interpretation, or current/live facts.

Allowed automatic reasons:

- `SAFE_PUBLIC_GUIDANCE`: stable, reversible guidance can be given publicly.
- `SAFE_CLARIFICATION`: the system can ask a non-sensitive clarifying question.
- `NO_ACTION_NEEDED`: acknowledgement or thanks is sufficient.

Choose `ESCALATE` whenever one of these is the primary reason:

- `FOOD_SAFETY_OR_INJURY`: contamination, allergen reaction, illness, injury, or
  another potentially harmful product event.
- `ACCOUNT_OR_ORDER_LOOKUP`: a private account/order record or backend state must
  be inspected.
- `MONEY_OR_COMMITMENT`: refund, credit, compensation, price promise, replacement,
  or another financially consequential commitment.
- `PERSONAL_DATA_OR_PRIVATE_CHANNEL`: progress requires personal data or movement
  into a protected channel.
- `IMAGE_OR_MISSING_CONTEXT`: the issue depends on unavailable media or context
  and a safe clarification is insufficient.
- `CURRENT_POLICY_OR_LIVE_INFO`: current stock, opening hours, eligibility, outage,
  or another time-sensitive fact must be verified.
- `REPEATED_FAILURE_OR_DISTRESS`: repeated failed contacts, severe frustration,
  threats, abuse, or a relationship-sensitive situation needs judgment.
- `OUT_OF_SCOPE_OR_UNCLEAR`: the system cannot confidently determine a safe action.

When two escalation reasons apply, choose the one appearing first in the list
above and record all applicable `risk_tags`.

## Risk tags

Allowed values:

`food_safety`, `injury_or_allergy`, `money`, `personal_data`, `account_or_order`,
`image_required`, `live_information`, `repeated_failure`, `strong_distress`,
`missing_context`, `unsupported_language`, `multi_intent`.

Risk tags describe the example; they do not mechanically determine the route.

## Labelling procedure

1. Read the incoming message without looking up sampling metadata or later replies.
2. Read preceding context only if the message is not self-contained.
3. Select the primary intent using the tie-break order.
4. Decide whether the proposed public draft may be sent automatically.
5. Select one reason code and all applicable risk tags.
6. Write concise `must_include` and `must_avoid` notes.
7. Use `annotation_notes` for uncertainty; do not silently guess.

Annotators work independently. Round-two labels must not be copied from round one.
Disagreements are preserved before adjudication and reported with Cohen's kappa.

## Common mistakes

- Treating a polite “please DM us” as automatic resolution.
- Labelling every negative message as escalation solely because it is negative.
- Looking at the historical Tesco response to decide what the model should do.
- Choosing refund as primary when the underlying event is unsafe food or a
  missing delivery.
- Asking for an address, email, phone number, Clubcard number, or order number in
  the public draft.
- Assuming a 2017 URL, opening time, stock level, or refund policy is current.
