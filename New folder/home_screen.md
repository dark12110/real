# Home Screen - Atmosphere Attire Advisor (A3)

```
┌─────────────────────────────────────────┐
│                                         │
│  ☰                               👤     │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  23°C ☀️ Nairobi                │    │
│  │  Sunny with light breeze        │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Today's Recommendations                 │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ ┌─────────┐ ┌─────────┐ ┌─────┐ │    │
│  │ │         │ │         │ │     │ │    │
│  │ │  Top    │ │ Bottom  │ │Shoes│ │    │
│  │ │         │ │         │ │     │ │    │
│  │ └─────────┘ └─────────┘ └─────┘ │    │
│  │                                 │    │
│  │         Casual Outfit           │    │
│  │                                 │    │
│  │      ┌───────────────┐          │    │
│  │      │   Wear This   │          │    │
│  │      └───────────────┘          │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Occasion:  ┌─────────────────┐         │
│             │ Casual      ▼   │         │
│             └─────────────────┘         │
│                                         │
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
- Menu icon (hamburger) on the left
- Profile icon on the right

### Weather Widget
- Current temperature and weather condition icon
- Location name
- Brief weather description

### Today's Recommendations
- Card showing outfit recommendation
- Images of clothing items (top, bottom, shoes)
- Outfit style label
- "Wear This" button
- Swipeable for more recommendations (not shown in static mockup)

### Occasion Selector
- Dropdown menu for selecting occasion type
- Default set to "Casual"
- Options include: Formal, Business, Casual, Sports, Cultural

### Floating Action Button
- "+" icon for adding new clothing items (not shown in static mockup)
- Located at bottom right, above the navigation bar

### Bottom Navigation
- Home (active)
- Wardrobe
- Stats (Analytics)
- Profile

## Design Notes:

1. **Weather Widget Interaction:**
   - Tapping refreshes weather data
   - Long press opens location selection

2. **Recommendation Card Interaction:**
   - Swipe horizontally to view more outfit options
   - Tap on individual clothing items to see details
   - "Wear This" button logs the selection and offers to save to calendar

3. **Accessibility Features:**
   - Weather information announced by screen readers
   - Swipe gestures have alternative button controls
   - High contrast mode available

4. **Responsive Behavior:**
   - On larger screens, can show multiple outfit recommendations side by side
   - Weather widget maintains fixed height but expands horizontally

5. **Animation:**
   - Subtle parallax effect when scrolling through recommendations
   - Weather refresh has a spinning animation
   - Smooth transitions when changing occasions
