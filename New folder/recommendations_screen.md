# Recommendations Engine Screen - Atmosphere Attire Advisor (A3)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Recommendations                      │
│                                         │
│  Occasion:  ┌─────────────────┐         │
│             │ Formal      ▼   │         │
│             └─────────────────┘         │
│                                         │
│  Weather: 23°C ☀️ Nairobi               │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  ┌─────┐    ┌─────┐    ┌─────┐  │    │
│  │  │     │    │     │    │     │  │    │
│  │  │  👔  │    │  👖  │    │  👞  │  │    │
│  │  │     │    │     │    │     │  │    │
│  │  └─────┘    └─────┘    └─────┘  │    │
│  │                                 │    │
│  │  White Shirt   Black Pants   Oxfords │    │
│  │                                 │    │
│  │  ✔️ Suitable for current weather │    │
│  │                                 │    │
│  │  ┌─────┐    ┌─────┐    ┌─────┐  │    │
│  │  │ ❤️  │    │ 💔  │    │ ⏱️  │  │    │
│  │  │Like │    │Dislike│   │Later│  │    │
│  │  └─────┘    └─────┘    └─────┘  │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  ┌─────┐    ┌─────┐    ┌─────┐  │    │
│  │  │     │    │     │    │     │  │    │
│  │  │  👔  │    │  👖  │    │  👞  │  │    │
│  │  │     │    │     │    │     │  │    │
│  │  └─────┘    └─────┘    └─────┘  │    │
│  │                                 │    │
│  │  Blue Shirt    Navy Pants    Loafers │    │
│  │                                 │    │
│  │  ✔️ Suitable for current weather │    │
│  │                                 │    │
│  │  ┌─────┐    ┌─────┐    ┌─────┐  │    │
│  │  │ ❤️  │    │ 💔  │    │ ⏱️  │  │    │
│  │  │Like │    │Dislike│   │Later│  │    │
│  │  └─────┘    └─────┘    └─────┘  │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│  │     │     │     │     │     │     │     │
│  │ 🏠  │     │ 👕  │     │ 📊  │     │ 👤  │
│  │Home │     │Ward-│     │Stats│     │Prof-│
│  │     │     │robe │     │     │     │ile  │
│  └─────┘     └─────┘     └─────┘     └─────┘
└─────────────────────────────────────────┘
```

## Components:

### App Bar
- Back button with title "Recommendations"

### Occasion Selector
- Dropdown menu for selecting occasion type
- Options: Formal, Business, Casual, Sports, Cultural
- Currently set to "Formal"

### Weather Information
- Current temperature, condition icon, and location
- Used as context for recommendations

### Recommendation Cards
- Scrollable list of outfit recommendations
- Each card contains:
  - Images of clothing items (top, bottom, shoes)
  - Names of each item
  - Weather compatibility badge
  - Action buttons: Like, Dislike, Save for Later

## Feedback Dialog (appears after Like/Dislike)

```
┌─────────────────────────────────────────┐
│                                         │
│         Rate This Recommendation         │
│                                         │
│  ⭐⭐⭐⭐⭐                               │
│                                         │
│  What did you think?                    │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│         ┌───────────────┐               │
│         │    Submit     │               │
│         └───────────────┘               │
│                                         │
│              Skip                       │
│                                         │
└─────────────────────────────────────────┘
```

## Cultural Context Settings (accessible from menu)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Cultural Preferences                 │
│                                         │
│  Region-Specific Styles                 │
│                                         │
│  ○ Standard                             │
│  ● Modest Wear                          │
│  ○ Traditional                          │
│  ○ Custom                               │
│                                         │
│  Custom Settings                        │
│                                         │
│  ☑ Cover shoulders                      │
│  ☑ Below-knee length                    │
│  ☐ Cover head                           │
│  ☐ Avoid tight-fitting clothes          │
│                                         │
│  Special Occasions                      │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ Add cultural occasion...         │    │
│  └─────────────────────────────────┘    │
│                                         │
│         ┌───────────────┐               │
│         │     Save      │               │
│         └───────────────┘               │
│                                         │
└─────────────────────────────────────────┘
```

## Design Notes:

1. **Recommendation Algorithm Visualization:**
   - Weather factors (temperature, precipitation, wind)
   - Occasion appropriateness
   - User style preferences
   - Previous feedback

2. **Interaction Design:**
   - Swipe left to dislike
   - Swipe right to like
   - Swipe up to save for later
   - Tap on individual items to see alternatives

3. **Personalization Features:**
   - Learning from user feedback
   - Adapting to seasonal changes
   - Considering color preferences
   - Respecting cultural contexts

4. **Accessibility:**
   - Alternative text for all clothing items
   - Non-visual feedback for likes/dislikes
   - Voice control for navigating recommendations

5. **Performance Optimization:**
   - Lazy loading of recommendation cards
   - Caching frequent recommendations
   - Background processing of new recommendations
