# Onboarding Screens - Atmosphere Attire Advisor (A3)

## Welcome Screen

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│                                         │
│               [App Logo]                │
│                                         │
│         Atmosphere Attire Advisor       │
│                                         │
│       Your personal wardrobe stylist    │
│                                         │
│                                         │
│                                         │
│                                         │
│         ┌───────────────────┐          │
│         │    Get Started    │          │
│         └───────────────────┘          │
│                                         │
│                                         │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

### Components:
- App Logo (centered)
- App Name: "Atmosphere Attire Advisor"
- Tagline: "Your personal wardrobe stylist"
- Primary Button: "Get Started"
- Material Design 3 styling with dynamic color support

## Permissions Screen 1: Camera

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Back                              1/3 │
│                                         │
│                                         │
│               [Camera Icon]             │
│                                         │
│           Camera Permission             │
│                                         │
│  We need access to your camera to help  │
│  you digitize your wardrobe items and   │
│  create your personalized collection.   │
│                                         │
│                                         │
│         ┌───────────────────┐          │
│         │  Allow Camera     │          │
│         └───────────────────┘          │
│                                         │
│               Skip for now              │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

### Components:
- Back button (top left)
- Progress indicator: "1/3" (top right)
- Camera icon (centered)
- Title: "Camera Permission"
- Description text explaining why camera access is needed
- Primary Button: "Allow Camera"
- Text Button: "Skip for now"

## Permissions Screen 2: Location

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Back                              2/3 │
│                                         │
│                                         │
│             [Location Icon]             │
│                                         │
│          Location Permission            │
│                                         │
│  We need your location to provide       │
│  weather-appropriate outfit             │
│  recommendations for your area.         │
│                                         │
│                                         │
│         ┌───────────────────┐          │
│         │  Allow Location   │          │
│         └───────────────────┘          │
│                                         │
│               Skip for now              │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

### Components:
- Back button (top left)
- Progress indicator: "2/3" (top right)
- Location icon (centered)
- Title: "Location Permission"
- Description text explaining why location access is needed
- Primary Button: "Allow Location"
- Text Button: "Skip for now"

## Permissions Screen 3: Storage

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Back                              3/3 │
│                                         │
│                                         │
│              [Storage Icon]             │
│                                         │
│           Storage Permission            │
│                                         │
│  We need storage access to save your    │
│  wardrobe images and outfit             │
│  recommendations.                       │
│                                         │
│                                         │
│         ┌───────────────────┐          │
│         │  Allow Storage    │          │
│         └───────────────────┘          │
│                                         │
│               Skip for now              │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

### Components:
- Back button (top left)
- Progress indicator: "3/3" (top right)
- Storage icon (centered)
- Title: "Storage Permission"
- Description text explaining why storage access is needed
- Primary Button: "Allow Storage"
- Text Button: "Skip for now"

## Wardrobe Setup Guide

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Back                                 │
│                                         │
│       Let's Set Up Your Wardrobe        │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │                                 │    │
│  │       [Wardrobe Setup           │    │
│  │        Illustration]            │    │
│  │                                 │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Take photos of your clothing items     │
│  to build your digital wardrobe.        │
│                                         │
│         ┌───────────────────┐          │
│         │  Start Adding     │          │
│         └───────────────────┘          │
│                                         │
│               Do it later               │
│                                         │
└─────────────────────────────────────────┘
```

### Components:
- Back button (top left)
- Title: "Let's Set Up Your Wardrobe"
- Illustration showing wardrobe setup process
- Description text explaining the process
- Primary Button: "Start Adding"
- Text Button: "Do it later"

## Design Notes:

1. **Color Scheme:**
   - Primary: #006064 (Deep Teal)
   - Secondary: #FF6E40 (Coral)
   - Background: #FFFFFF (White)
   - Surface: #F5F5F5 (Light Gray)
   - Text: #212121 (Dark Gray)

2. **Typography:**
   - Headings: Roboto Medium
   - Body: Roboto Regular
   - Buttons: Roboto Medium

3. **Animations:**
   - Subtle fade transitions between screens
   - Button ripple effects
   - Progress indicator animation

4. **Accessibility:**
   - High contrast text
   - Adequate touch target sizes (minimum 48dp)
   - Content descriptions for all images
   - Support for screen readers
