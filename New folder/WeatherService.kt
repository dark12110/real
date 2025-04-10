package com.example.a3app.data.weather

import retrofit2.http.GET
import retrofit2.http.Query

/**
 * Weather API service interface for the A3 app
 */
interface WeatherApiService {
    @GET("weather")
    suspend fun getCurrentWeather(
        @Query("lat") latitude: Double,
        @Query("lon") longitude: Double,
        @Query("units") units: String = "metric",
        @Query("appid") apiKey: String
    ): WeatherResponse
    
    @GET("forecast")
    suspend fun getForecast(
        @Query("lat") latitude: Double,
        @Query("lon") longitude: Double,
        @Query("units") units: String = "metric",
        @Query("appid") apiKey: String
    ): ForecastResponse
}

/**
 * Weather response data classes
 */
data class WeatherResponse(
    val main: Main,
    val weather: List<Weather>,
    val name: String,
    val wind: Wind
)

data class ForecastResponse(
    val list: List<ForecastItem>
)

data class ForecastItem(
    val dt: Long,
    val main: Main,
    val weather: List<Weather>,
    val wind: Wind
)

data class Main(
    val temp: Double,
    val feels_like: Double,
    val temp_min: Double,
    val temp_max: Double,
    val humidity: Int
)

data class Weather(
    val id: Int,
    val main: String,
    val description: String,
    val icon: String
)

data class Wind(
    val speed: Double,
    val deg: Int
)

/**
 * Weather repository for the A3 app
 */
class WeatherRepository(private val weatherApiService: WeatherApiService) {
    
    suspend fun getCurrentWeather(latitude: Double, longitude: Double, apiKey: String): WeatherResponse {
        return weatherApiService.getCurrentWeather(latitude, longitude, apiKey = apiKey)
    }
    
    suspend fun getForecast(latitude: Double, longitude: Double, apiKey: String): ForecastResponse {
        return weatherApiService.getForecast(latitude, longitude, apiKey = apiKey)
    }
    
    /**
     * Maps weather conditions to clothing recommendations
     */
    fun getClothingRecommendations(weatherResponse: WeatherResponse): ClothingRecommendations {
        val temperature = weatherResponse.main.temp
        val weatherCondition = weatherResponse.weather.firstOrNull()?.main ?: "Clear"
        val windSpeed = weatherResponse.wind.speed
        
        return when {
            // Hot weather
            temperature > 30 -> ClothingRecommendations(
                tops = listOf("T-shirt", "Tank top"),
                bottoms = listOf("Shorts", "Light pants"),
                outerwear = emptyList(),
                accessories = listOf("Sunglasses", "Hat")
            )
            
            // Warm weather
            temperature in 20.0..30.0 -> ClothingRecommendations(
                tops = listOf("T-shirt", "Short sleeve shirt"),
                bottoms = listOf("Shorts", "Light pants", "Skirt"),
                outerwear = if (temperature < 25) listOf("Light jacket") else emptyList(),
                accessories = listOf("Sunglasses")
            )
            
            // Mild weather
            temperature in 15.0..20.0 -> ClothingRecommendations(
                tops = listOf("Long sleeve shirt", "Light sweater"),
                bottoms = listOf("Pants", "Jeans"),
                outerwear = listOf("Light jacket"),
                accessories = emptyList()
            )
            
            // Cool weather
            temperature in 10.0..15.0 -> ClothingRecommendations(
                tops = listOf("Long sleeve shirt", "Sweater"),
                bottoms = listOf("Pants", "Jeans"),
                outerwear = listOf("Jacket", "Light coat"),
                accessories = listOf("Light scarf")
            )
            
            // Cold weather
            temperature in 0.0..10.0 -> ClothingRecommendations(
                tops = listOf("Sweater", "Thermal shirt"),
                bottoms = listOf("Jeans", "Thick pants"),
                outerwear = listOf("Coat", "Jacket"),
                accessories = listOf("Scarf", "Gloves")
            )
            
            // Very cold weather
            else -> ClothingRecommendations(
                tops = listOf("Thermal shirt", "Sweater"),
                bottoms = listOf("Thermal pants", "Thick pants"),
                outerwear = listOf("Heavy coat", "Parka"),
                accessories = listOf("Scarf", "Gloves", "Hat")
            )
        }.let { recommendations ->
            // Adjust for rain
            if (weatherCondition == "Rain" || weatherCondition == "Drizzle") {
                recommendations.copy(
                    outerwear = recommendations.outerwear + "Raincoat",
                    accessories = recommendations.accessories + "Umbrella",
                    footwear = listOf("Waterproof shoes", "Boots")
                )
            } 
            // Adjust for snow
            else if (weatherCondition == "Snow") {
                recommendations.copy(
                    outerwear = recommendations.outerwear + "Waterproof coat",
                    accessories = recommendations.accessories + "Warm hat",
                    footwear = listOf("Snow boots", "Waterproof boots")
                )
            } 
            // Adjust for wind
            else if (windSpeed > 5.5) {
                recommendations.copy(
                    outerwear = recommendations.outerwear + "Windbreaker",
                    accessories = recommendations.accessories + "Hat"
                )
            } else {
                recommendations
            }
        }
    }
}

/**
 * Clothing recommendations based on weather
 */
data class ClothingRecommendations(
    val tops: List<String>,
    val bottoms: List<String>,
    val outerwear: List<String>,
    val accessories: List<String>,
    val footwear: List<String> = listOf("Shoes", "Sneakers")
)
