# User Profile Screen - Atmosphere Attire Advisor (A3)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Profile                        ⚙️    │
│                                         │
│         ┌─────────────────┐             │
│         │                 │             │
│         │       👤        │             │
│         │                 │             │
│         └─────────────────┘             │
│                                         │
│  Sarah Johnson                          │
│  sarah.johnson@email.com                │
│                                         │
│  Style Preferences                      │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  Preferred Styles:              │    │
│  │  ┌─────┐  ┌─────┐  ┌─────┐      │    │
│  │  │Casual│  │Formal│  │Sport │    │    │
│  │  └─────┘  └─────┘  └─────┘      │    │
│  │                                 │    │
│  │  Color Palette:                 │    │
│  │  ┌───┐┌───┐┌───┐┌───┐┌───┐      │    │
│  │  │🔵 ││⚫ ││🔴 ││⚪ ││🟢 │      │    │
│  │  └───┘└───┘└───┘└───┘└───┘      │    │
│  │                                 │    │
│  │  Avoid: Neon colors, Patterns   │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Notification Settings                  │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  ☑ Weather alerts               │    │
│  │  ☑ Wardrobe suggestions         │    │
│  │  ☐ Unworn item reminders        │    │
│  │  ☑ Special occasion reminders   │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Account                               │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  ☑ Sync with Google Drive       │    │
│  │  ☐ Dark mode                    │    │
│  │                                 │    │
│  │  ┌───────────────────┐          │    │
│  │  │  Backup Wardrobe  │          │    │
│  │  └───────────────────┘          │    │
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
- Back button with title "Profile"
- Settings icon for advanced settings

### Profile Header
- User avatar/photo
- User name
- Email address

### Style Preferences Section
- Preferred style chips (Casual, Formal, Sport)
- Color palette swatches
- Avoid preferences

### Notification Settings Section
- Checkboxes for different notification types:
  - Weather alerts
  - Wardrobe suggestions
  - Unworn item reminders
  - Special occasion reminders

### Account Section
- Sync options
- Theme toggle (Dark mode)
- Backup button

## Settings Screen (when settings icon is tapped)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Settings                             │
│                                         │
│  App Settings                           │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  Language                       │    │
│  │  ┌─────────────────┐            │    │
│  │  │ English      ▼  │            │    │
│  │  └─────────────────┘            │    │
│  │                                 │    │
│  │  Temperature Unit               │    │
│  │  ○ Celsius    ● Fahrenheit      │    │
│  │                                 │    │
│  │  Start Screen                   │    │
│  │  ┌─────────────────┐            │    │
│  │  │ Home         ▼  │            │    │
│  │  └─────────────────┘            │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Privacy                                │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  Data Collection                │    │
│  │  ☑ Usage statistics             │    │
│  │  ☐ Personalized recommendations │    │
│  │                                 │    │
│  │  ┌───────────────────┐          │    │
│  │  │  Clear All Data   │          │    │
│  │  └───────────────────┘          │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  About                                  │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  Version: 1.0.0                 │    │
│  │                                 │    │
│  │  ┌───────────────────┐          │    │
│  │  │  Terms of Service │          │    │
│  │  └───────────────────┘          │    │
│  │                                 │    │
│  │  ┌───────────────────┐          │    │
│  │  │  Privacy Policy   │          │    │
│  │  └───────────────────┘          │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

## Style Preferences Editor (when Style Preferences is tapped)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Edit Style Preferences               │
│                                         │
│  Style Categories                       │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  ☑ Casual                       │    │
│  │  ☑ Formal                       │    │
│  │  ☑ Sport                        │    │
│  │  ☐ Business                     │    │
│  │  ☐ Evening                      │    │
│  │  ☐ Cultural                     │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Color Preferences                      │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  Favorite Colors:               │    │
│  │  ┌───┐┌───┐┌───┐┌───┐┌───┐      │    │
│  │  │🔵 ││⚫ ││🔴 ││⚪ ││🟢 │      │    │
│  │  └───┘└───┘└───┘└───┘└───┘      │    │
│  │                                 │    │
│  │  Colors to Avoid:               │    │
│  │  ┌───┐┌───┐┌───┐                │    │
│  │  │🟡 ││🟣 ││🟠 │                │    │
│  │  └───┘└───┘└───┘                │    │
│  │                                 │    │
│  │  ┌───────────────┐              │    │
│  │  │ Add Color   + │              │    │
│  │  └───────────────┘              │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Pattern Preferences                    │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │  ☐ Solid                        │    │
│  │  ☐ Stripes                      │    │
│  │  ☐ Checks                       │    │
│  │  ☑ Avoid patterns               │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│         ┌───────────────┐               │
│         │     Save      │               │
│         └───────────────┘               │
│                                         │
└─────────────────────────────────────────┘
```

## Design Notes:

1. **Profile Customization:**
   - Ability to upload profile photo
   - Editable style preferences
   - Customizable notification settings

2. **Style Preferences:**
   - Multiple selection of style categories
   - Color palette selection with visual swatches
   - Pattern preferences

3. **Notification Management:**
   - Granular control over notification types
   - Time-based settings (e.g., morning weather alerts)
   - Special occasion reminders

4. **Account Features:**
   - Cloud backup and sync
   - Theme customization
   - Data management

5. **Accessibility:**
   - High contrast mode
   - Text size adjustment
   - Screen reader support
   - Voice control options
