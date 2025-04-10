package com.example.a3app.data.storage

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/**
 * User preferences data store for the A3 app
 */
class UserPreferencesRepository(private val context: Context) {
    
    companion object {
        private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "user_preferences")
        
        // Preferences keys
        val DARK_MODE = booleanPreferencesKey("dark_mode")
        val TEMPERATURE_UNIT = stringPreferencesKey("temperature_unit")
        val LOCATION = stringPreferencesKey("location")
        val PREFERRED_STYLES = stringPreferencesKey("preferred_styles")
        val COLOR_PALETTE = stringPreferencesKey("color_palette")
        val AVOID_ITEMS = stringPreferencesKey("avoid_items")
        val WEATHER_ALERTS = booleanPreferencesKey("weather_alerts")
        val WARDROBE_SUGGESTIONS = booleanPreferencesKey("wardrobe_suggestions")
        val UNWORN_ITEM_REMINDERS = booleanPreferencesKey("unworn_item_reminders")
        val SPECIAL_OCCASION_REMINDERS = booleanPreferencesKey("special_occasion_reminders")
        val SYNC_WITH_GOOGLE_DRIVE = booleanPreferencesKey("sync_with_google_drive")
        val ONBOARDING_COMPLETED = booleanPreferencesKey("onboarding_completed")
    }
    
    // Get user preferences as a Flow
    val userPreferencesFlow: Flow<UserPreferences> = context.dataStore.data
        .map { preferences ->
            UserPreferences(
                darkMode = preferences[DARK_MODE] ?: false,
                temperatureUnit = preferences[TEMPERATURE_UNIT] ?: "celsius",
                location = preferences[LOCATION] ?: "",
                preferredStyles = preferences[PREFERRED_STYLES]?.split(",") ?: listOf("Casual", "Formal", "Sport"),
                colorPalette = preferences[COLOR_PALETTE]?.split(",") ?: listOf("Blue", "Black", "White"),
                avoidItems = preferences[AVOID_ITEMS] ?: "Neon colors, Patterns",
                weatherAlerts = preferences[WEATHER_ALERTS] ?: true,
                wardrobeSuggestions = preferences[WARDROBE_SUGGESTIONS] ?: true,
                unwornItemReminders = preferences[UNWORN_ITEM_REMINDERS] ?: false,
                specialOccasionReminders = preferences[SPECIAL_OCCASION_REMINDERS] ?: true,
                syncWithGoogleDrive = preferences[SYNC_WITH_GOOGLE_DRIVE] ?: false,
                onboardingCompleted = preferences[ONBOARDING_COMPLETED] ?: false
            )
        }
    
    // Update user preferences
    suspend fun updateUserPreferences(userPreferences: UserPreferences) {
        context.dataStore.edit { preferences ->
            preferences[DARK_MODE] = userPreferences.darkMode
            preferences[TEMPERATURE_UNIT] = userPreferences.temperatureUnit
            preferences[LOCATION] = userPreferences.location
            preferences[PREFERRED_STYLES] = userPreferences.preferredStyles.joinToString(",")
            preferences[COLOR_PALETTE] = userPreferences.colorPalette.joinToString(",")
            preferences[AVOID_ITEMS] = userPreferences.avoidItems
            preferences[WEATHER_ALERTS] = userPreferences.weatherAlerts
            preferences[WARDROBE_SUGGESTIONS] = userPreferences.wardrobeSuggestions
            preferences[UNWORN_ITEM_REMINDERS] = userPreferences.unwornItemReminders
            preferences[SPECIAL_OCCASION_REMINDERS] = userPreferences.specialOccasionReminders
            preferences[SYNC_WITH_GOOGLE_DRIVE] = userPreferences.syncWithGoogleDrive
            preferences[ONBOARDING_COMPLETED] = userPreferences.onboardingCompleted
        }
    }
    
    // Update a single preference
    suspend fun updateDarkMode(darkMode: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[DARK_MODE] = darkMode
        }
    }
    
    suspend fun updateTemperatureUnit(temperatureUnit: String) {
        context.dataStore.edit { preferences ->
            preferences[TEMPERATURE_UNIT] = temperatureUnit
        }
    }
    
    suspend fun updateLocation(location: String) {
        context.dataStore.edit { preferences ->
            preferences[LOCATION] = location
        }
    }
    
    suspend fun updatePreferredStyles(preferredStyles: List<String>) {
        context.dataStore.edit { preferences ->
            preferences[PREFERRED_STYLES] = preferredStyles.joinToString(",")
        }
    }
    
    suspend fun updateColorPalette(colorPalette: List<String>) {
        context.dataStore.edit { preferences ->
            preferences[COLOR_PALETTE] = colorPalette.joinToString(",")
        }
    }
    
    suspend fun updateAvoidItems(avoidItems: String) {
        context.dataStore.edit { preferences ->
            preferences[AVOID_ITEMS] = avoidItems
        }
    }
    
    suspend fun updateNotificationSettings(
        weatherAlerts: Boolean,
        wardrobeSuggestions: Boolean,
        unwornItemReminders: Boolean,
        specialOccasionReminders: Boolean
    ) {
        context.dataStore.edit { preferences ->
            preferences[WEATHER_ALERTS] = weatherAlerts
            preferences[WARDROBE_SUGGESTIONS] = wardrobeSuggestions
            preferences[UNWORN_ITEM_REMINDERS] = unwornItemReminders
            preferences[SPECIAL_OCCASION_REMINDERS] = specialOccasionReminders
        }
    }
    
    suspend fun updateSyncWithGoogleDrive(syncWithGoogleDrive: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[SYNC_WITH_GOOGLE_DRIVE] = syncWithGoogleDrive
        }
    }
    
    suspend fun completeOnboarding() {
        context.dataStore.edit { preferences ->
            preferences[ONBOARDING_COMPLETED] = true
        }
    }
}

/**
 * User preferences data class
 */
data class UserPreferences(
    val darkMode: Boolean,
    val temperatureUnit: String,
    val location: String,
    val preferredStyles: List<String>,
    val colorPalette: List<String>,
    val avoidItems: String,
    val weatherAlerts: Boolean,
    val wardrobeSuggestions: Boolean,
    val unwornItemReminders: Boolean,
    val specialOccasionReminders: Boolean,
    val syncWithGoogleDrive: Boolean,
    val onboardingCompleted: Boolean
)
