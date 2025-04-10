# Analytics Dashboard Screen - Atmosphere Attire Advisor (A3)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Analytics                      📅    │
│                                         │
│  Wardrobe Insights                      │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ Utilization Heatmap             │    │
│  │                                 │    │
│  │ ┌───┬───┬───┬───┬───┬───┬───┐   │    │
│  │ │   │███│███│   │   │███│███│   │    │
│  │ ├───┼───┼───┼───┼───┼───┼───┤   │    │
│  │ │███│   │   │███│███│   │   │   │    │
│  │ ├───┼───┼───┼───┼───┼───┼───┤   │    │
│  │ │███│███│   │   │   │███│███│   │    │
│  │ └───┴───┴───┴───┴───┴───┴───┘   │    │
│  │                                 │    │
│  │ Most worn: Blue jeans (15 times)│    │
│  │ Least worn: Green sweater (1)   │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Time Saved                             │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │ You've saved 2.1 hours this month    │
│  │                                 │    │
│  │ ┌───────────────────────────┐   │    │
│  │ │███████████████            │   │    │
│  │ └───────────────────────────┘   │    │
│  │                                 │    │
│  │ 63% reduction in decision time  │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│  Cost Per Wear                          │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │ ┌───┐  ┌───┐  ┌───┐  ┌───┐      │    │
│  │ │███│  │██ │  │█  │  │   │      │    │
│  │ │███│  │██ │  │█  │  │   │      │    │
│  │ │███│  │██ │  │█  │  │   │      │    │
│  │ │███│  │██ │  │█  │  │   │      │    │
│  │ └───┘  └───┘  └───┘  └───┘      │    │
│  │ $2.50   $5.00  $10.00 $20.00    │    │
│  │                                 │    │
│  │ Best value: Black t-shirt ($2.50/wear)│
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
- Back button with title "Analytics"
- Calendar icon for date range selection

### Wardrobe Insights Section
- Utilization Heatmap visualization
  - Shows frequency of item usage
  - Color intensity indicates wear frequency
- Most/least worn item statistics

### Time Saved Section
- Hours saved metric
- Progress bar visualization
- Percentage reduction in decision time

### Cost Per Wear Section
- Bar chart showing cost efficiency
- Items grouped by cost per wear ranges
- Best value item highlighted

## Calendar View (when calendar icon is tapped)

```
┌─────────────────────────────────────────┐
│                                         │
│  ← Calendar View                        │
│                                         │
│  April 2025                             │
│                                         │
│  ┌───┬───┬───┬───┬───┬───┬───┐         │
│  │Mon│Tue│Wed│Thu│Fri│Sat│Sun│         │
│  ├───┼───┼───┼───┼───┼───┼───┤         │
│  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │         │
│  ├───┼───┼───┼───┼───┼───┼───┤         │
│  │ 8 │ 9 │10 │11 │12 │13 │14 │         │
│  ├───┼───┼───┼───┼───┼───┼───┤         │
│  │15 │16 │17 │18 │19 │20 │21 │         │
│  ├───┼───┼───┼───┼───┼───┼───┤         │
│  │22 │23 │24 │25 │26 │27 │28 │         │
│  ├───┼───┼───┼───┼───┼───┼───┤         │
│  │29 │30 │   │   │   │   │   │         │
│  └───┴───┴───┴───┴───┴───┴───┘         │
│                                         │
│  April 15, 2025                         │
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
│  │  Occasion: Business Meeting    │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│         ┌───────────────┐               │
│         │    Export     │               │
│         └───────────────┘               │
│                                         │
└─────────────────────────────────────────┘
```

## Export Options Dialog (when Export is tapped)

```
┌─────────────────────────────────────────┐
│                                         │
│           Export Analytics              │
│                                         │
│  Select Format:                         │
│                                         │
│  ○ PDF Report                           │
│  ● CSV Data                             │
│  ○ Image Gallery                        │
│                                         │
│  Date Range:                            │
│                                         │
│  ┌─────────────────┐ ┌─────────────────┐│
│  │ 01/04/2025      │ │ 30/04/2025      ││
│  └─────────────────┘ └─────────────────┘│
│                                         │
│  Include:                               │
│                                         │
│  ☑ Utilization data                     │
│  ☑ Cost analysis                        │
│  ☑ Time savings                         │
│  ☐ Individual item details              │
│                                         │
│  ┌─────────────────┐ ┌─────────────────┐│
│  │     Cancel      │ │     Export      ││
│  └─────────────────┘ └─────────────────┘│
│                                         │
└─────────────────────────────────────────┘
```

## Design Notes:

1. **Data Visualization:**
   - Interactive heatmap (tap to see details)
   - Animated progress bars for time saved
   - Color-coded cost efficiency chart

2. **Time Period Selection:**
   - Default view shows current month
   - Options for week, month, year views
   - Custom date range selection

3. **Export Functionality:**
   - PDF report generation
   - CSV data export
   - Image gallery of worn outfits

4. **Accessibility Features:**
   - Alternative text descriptions for charts
   - High contrast mode for visualizations
   - Numerical data alongside visual representations

5. **Performance Considerations:**
   - Lazy loading of chart data
   - Caching of frequently accessed statistics
   - Background calculation of complex metrics
