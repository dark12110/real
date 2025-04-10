# Wardrobe Management Screen - Atmosphere Attire Advisor (A3)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← My Wardrobe                    🔍    │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ All Items (42) ▼                │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│
│  │Tops │ │Botto│ │Dress│ │Outer│ │Shoes││
│  │(15) │ │ms(12│ │es(5)│ │(6)  │ │(4)  ││
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘│
│                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
│  │     │ │     │ │     │ │     │        │
│  │  👕  │ │  👕  │ │  👕  │ │  👕  │        │
│  │     │ │     │ │     │ │     │        │
│  └─────┘ └─────┘ └─────┘ └─────┘        │
│                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
│  │     │ │     │ │     │ │     │        │
│  │  👕  │ │  👕  │ │  👕  │ │  👕  │        │
│  │     │ │     │ │     │ │     │        │
│  └─────┘ └─────┘ └─────┘ └─────┘        │
│                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐        │
│  │     │ │     │ │     │ │     │        │
│  │  👕  │ │  👕  │ │  👕  │ │  👕  │        │
│  │     │ │     │ │     │ │     │        │
│  └─────┘ └─────┘ └─────┘ └─────┘        │
│                                         │
│                  ⊕                      │
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
- Back button with title "My Wardrobe"
- Search icon for filtering items

### Category Filter
- Dropdown showing current filter ("All Items" with count)
- Horizontal scrollable category chips:
  - Tops (15)
  - Bottoms (12)
  - Dresses (5)
  - Outerwear (6)
  - Shoes (4)
  - (More categories available by scrolling)

### Wardrobe Grid
- Grid view of clothing items
- 4 columns on standard phone screens (responsive)
- Each item shows thumbnail image
- Tappable to view details

### Floating Action Button
- "+" icon for adding new clothing items
- Located at bottom center

### Bottom Navigation
- Home
- Wardrobe (active)
- Stats (Analytics)
- Profile

## Item Detail Screen (when item is tapped)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Back                          ⋮      │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │                                 │    │
│  │                                 │    │
│  │           [Clothing             │    │
│  │            Image]               │    │
│  │                                 │    │
│  │                                 │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Blue Cotton T-shirt                    │
│  Casual                                 │
│                                         │
│  Details                                │
│  ┌─────────────────────────────────┐    │
│  │ Type:       T-shirt             │    │
│  │ Color:      Blue                │    │
│  │ Material:   Cotton              │    │
│  │ Style:      Casual              │    │
│  │ Last worn:  2 weeks ago         │    │
│  │ Times worn: 8                   │    │
│  └─────────────────────────────────┘    │
│                                         │
│         ┌───────────────┐               │
│         │     Edit      │               │
│         └───────────────┘               │
│                                         │
└─────────────────────────────────────────┘
```

## Add Item Screen (when FAB is tapped)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Cancel                      Done     │
│                                         │
│  Add Clothing Item                      │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │                                 │    │
│  │                                 │    │
│  │         [Camera Preview         │    │
│  │          or Image Upload]       │    │
│  │                                 │    │
│  │                                 │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  ┌─────────────────┐ ┌─────────────────┐│
│  │   Take Photo    │ │  Choose Photo   ││
│  └─────────────────┘ └─────────────────┘│
│                                         │
│  Item Details                           │
│  ┌─────────────────────────────────┐    │
│  │ Type:     ┌──────────────┐      │    │
│  │           │ Top       ▼  │      │    │
│  │           └──────────────┘      │    │
│  │                                 │    │
│  │ Color:    ┌──────────────┐      │    │
│  │           │ Blue      ▼  │      │    │
│  │           └──────────────┘      │    │
│  │                                 │    │
│  │ Material: ┌──────────────┐      │    │
│  │           │ Cotton    ▼  │      │    │
│  │           └──────────────┘      │    │
│  │                                 │    │
│  │ Style:    ┌──────────────┐      │    │
│  │           │ Casual    ▼  │      │    │
│  │           └──────────────┘      │    │
│  └─────────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

## Design Notes:

1. **Grid Interaction:**
   - Tap on item opens detail view
   - Long press enables multi-select mode
   - Pull to refresh updates the wardrobe

2. **Filtering Options:**
   - Category filter (Tops, Bottoms, etc.)
   - Color filter
   - Style filter (Casual, Formal, etc.)
   - Season filter (Summer, Winter, etc.)
   - Recently added/worn

3. **Add Item Flow:**
   - Camera opens with auto-focus
   - Background removal happens automatically
   - AI suggests item details (type, color, material)
   - User can adjust/confirm details

4. **Batch Upload:**
   - Option to add multiple items at once
   - Progress indicator shows processing status
   - AI auto-tagging for efficient categorization

5. **Accessibility:**
   - Grid items have descriptive labels
   - Camera has alternative manual entry option
   - Color selection includes color names (not just visual swatches)
