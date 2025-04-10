package com.example.a3app.components

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.a3app.navigation.Screen

/**
 * Bottom navigation component for the A3 app
 */
@Composable
fun A3BottomNavigation(
    currentRoute: String,
    onNavigateToHome: () -> Unit,
    onNavigateToWardrobe: () -> Unit,
    onNavigateToAnalytics: () -> Unit,
    onNavigateToProfile: () -> Unit,
    modifier: Modifier = Modifier
) {
    NavigationBar(
        modifier = modifier
            .fillMaxWidth()
            .height(64.dp),
        containerColor = MaterialTheme.colorScheme.surface,
    ) {
        NavigationBarItem(
            icon = { 
                Icon(
                    painter = painterResource(id = android.R.drawable.ic_menu_home),
                    contentDescription = "Home"
                )
            },
            label = { 
                Text(
                    text = "Home",
                    style = MaterialTheme.typography.labelMedium,
                    textAlign = TextAlign.Center
                )
            },
            selected = currentRoute == Screen.Home.route,
            onClick = onNavigateToHome,
            colors = NavigationBarItemDefaults.colors(
                selectedIconColor = MaterialTheme.colorScheme.primary,
                selectedTextColor = MaterialTheme.colorScheme.primary,
                indicatorColor = MaterialTheme.colorScheme.surface
            )
        )
        
        NavigationBarItem(
            icon = { 
                Icon(
                    painter = painterResource(id = android.R.drawable.ic_menu_gallery),
                    contentDescription = "Wardrobe"
                )
            },
            label = { 
                Text(
                    text = "Wardrobe",
                    style = MaterialTheme.typography.labelMedium,
                    textAlign = TextAlign.Center
                )
            },
            selected = currentRoute == Screen.Wardrobe.route,
            onClick = onNavigateToWardrobe,
            colors = NavigationBarItemDefaults.colors(
                selectedIconColor = MaterialTheme.colorScheme.primary,
                selectedTextColor = MaterialTheme.colorScheme.primary,
                indicatorColor = MaterialTheme.colorScheme.surface
            )
        )
        
        NavigationBarItem(
            icon = { 
                Icon(
                    painter = painterResource(id = android.R.drawable.ic_menu_recent_history),
                    contentDescription = "Analytics"
                )
            },
            label = { 
                Text(
                    text = "Stats",
                    style = MaterialTheme.typography.labelMedium,
                    textAlign = TextAlign.Center
                )
            },
            selected = currentRoute == Screen.Analytics.route,
            onClick = onNavigateToAnalytics,
            colors = NavigationBarItemDefaults.colors(
                selectedIconColor = MaterialTheme.colorScheme.primary,
                selectedTextColor = MaterialTheme.colorScheme.primary,
                indicatorColor = MaterialTheme.colorScheme.surface
            )
        )
        
        NavigationBarItem(
            icon = { 
                Icon(
                    painter = painterResource(id = android.R.drawable.ic_menu_myplaces),
                    contentDescription = "Profile"
                )
            },
            label = { 
                Text(
                    text = "Profile",
                    style = MaterialTheme.typography.labelMedium,
                    textAlign = TextAlign.Center
                )
            },
            selected = currentRoute == Screen.Profile.route,
            onClick = onNavigateToProfile,
            colors = NavigationBarItemDefaults.colors(
                selectedIconColor = MaterialTheme.colorScheme.primary,
                selectedTextColor = MaterialTheme.colorScheme.primary,
                indicatorColor = MaterialTheme.colorScheme.surface
            )
        )
    }
}
