package com.example.a3app.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.a3app.screens.analytics.AnalyticsScreen
import com.example.a3app.screens.home.HomeScreen
import com.example.a3app.screens.onboarding.OnboardingScreen
import com.example.a3app.screens.profile.ProfileScreen
import com.example.a3app.screens.recommendations.RecommendationsScreen
import com.example.a3app.screens.wardrobe.WardrobeScreen

/**
 * Navigation routes for the A3 app
 */
sealed class Screen(val route: String) {
    object Onboarding : Screen("onboarding")
    object Home : Screen("home")
    object Wardrobe : Screen("wardrobe")
    object Recommendations : Screen("recommendations")
    object Analytics : Screen("analytics")
    object Profile : Screen("profile")
    
    // Detail screens
    object ItemDetail : Screen("item_detail/{itemId}") {
        fun createRoute(itemId: String) = "item_detail/$itemId"
    }
    
    object AddItem : Screen("add_item")
    object Settings : Screen("settings")
}

/**
 * Main navigation component for the A3 app
 */
@Composable
fun A3AppNavHost(
    navController: NavHostController,
    startDestination: String = Screen.Onboarding.route
) {
    NavHost(
        navController = navController,
        startDestination = startDestination
    ) {
        composable(Screen.Onboarding.route) {
            OnboardingScreen(
                onGetStartedClick = {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Onboarding.route) { inclusive = true }
                    }
                }
            )
        }
        
        composable(Screen.Home.route) {
            HomeScreen(
                onNavigateToWardrobe = { navController.navigate(Screen.Wardrobe.route) },
                onNavigateToRecommendations = { navController.navigate(Screen.Recommendations.route) },
                onNavigateToAnalytics = { navController.navigate(Screen.Analytics.route) },
                onNavigateToProfile = { navController.navigate(Screen.Profile.route) }
            )
        }
        
        composable(Screen.Wardrobe.route) {
            WardrobeScreen(
                onNavigateToHome = { navController.navigate(Screen.Home.route) },
                onNavigateToAnalytics = { navController.navigate(Screen.Analytics.route) },
                onNavigateToProfile = { navController.navigate(Screen.Profile.route) },
                onNavigateToItemDetail = { itemId ->
                    navController.navigate(Screen.ItemDetail.createRoute(itemId))
                },
                onNavigateToAddItem = { navController.navigate(Screen.AddItem.route) }
            )
        }
        
        composable(Screen.Recommendations.route) {
            RecommendationsScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToHome = { navController.navigate(Screen.Home.route) },
                onNavigateToWardrobe = { navController.navigate(Screen.Wardrobe.route) },
                onNavigateToAnalytics = { navController.navigate(Screen.Analytics.route) },
                onNavigateToProfile = { navController.navigate(Screen.Profile.route) }
            )
        }
        
        composable(Screen.Analytics.route) {
            AnalyticsScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToHome = { navController.navigate(Screen.Home.route) },
                onNavigateToWardrobe = { navController.navigate(Screen.Wardrobe.route) },
                onNavigateToProfile = { navController.navigate(Screen.Profile.route) }
            )
        }
        
        composable(Screen.Profile.route) {
            ProfileScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToHome = { navController.navigate(Screen.Home.route) },
                onNavigateToWardrobe = { navController.navigate(Screen.Wardrobe.route) },
                onNavigateToAnalytics = { navController.navigate(Screen.Analytics.route) },
                onNavigateToSettings = { navController.navigate(Screen.Settings.route) }
            )
        }
        
        // Additional screens can be added here
    }
}
