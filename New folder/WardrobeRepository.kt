package com.example.a3app.data.wardrobe

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Delete
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverter
import androidx.room.TypeConverters
import androidx.room.Update
import kotlinx.coroutines.flow.Flow
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Database for storing wardrobe items
 */
@Database(entities = [ClothingItem::class, OutfitHistory::class], version = 1, exportSchema = false)
@TypeConverters(Converters::class)
abstract class WardrobeDatabase : RoomDatabase() {
    abstract fun clothingItemDao(): ClothingItemDao
    abstract fun outfitHistoryDao(): OutfitHistoryDao
    
    companion object {
        @Volatile
        private var INSTANCE: WardrobeDatabase? = null
        
        fun getDatabase(context: Context): WardrobeDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    WardrobeDatabase::class.java,
                    "wardrobe_database"
                )
                .fallbackToDestructiveMigration()
                .build()
                INSTANCE = instance
                instance
            }
        }
    }
}

/**
 * Type converters for Room database
 */
class Converters {
    private val formatter = DateTimeFormatter.ISO_LOCAL_DATE_TIME
    
    @TypeConverter
    fun fromTimestamp(value: String?): LocalDateTime? {
        return value?.let { LocalDateTime.parse(it, formatter) }
    }
    
    @TypeConverter
    fun dateToTimestamp(date: LocalDateTime?): String? {
        return date?.format(formatter)
    }
}

/**
 * Entity representing a clothing item
 */
@Entity(tableName = "clothing_items")
data class ClothingItem(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val name: String,
    val type: String,
    val color: String,
    val material: String,
    val style: String,
    val imageUri: String,
    val lastWorn: LocalDateTime? = null,
    val timesWorn: Int = 0,
    val dateAdded: LocalDateTime = LocalDateTime.now()
)

/**
 * Entity representing outfit history
 */
@Entity(tableName = "outfit_history")
data class OutfitHistory(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val date: LocalDateTime = LocalDateTime.now(),
    val occasion: String,
    val topId: Long,
    val bottomId: Long,
    val shoesId: Long,
    val outerwearId: Long? = null,
    val accessoryIds: String? = null, // Comma-separated list of accessory IDs
    val rating: Int? = null,
    val feedback: String? = null,
    val weatherCondition: String? = null,
    val temperature: Double? = null
)

/**
 * DAO for clothing items
 */
@Dao
interface ClothingItemDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(clothingItem: ClothingItem): Long
    
    @Update
    suspend fun update(clothingItem: ClothingItem)
    
    @Delete
    suspend fun delete(clothingItem: ClothingItem)
    
    @Query("SELECT * FROM clothing_items ORDER BY dateAdded DESC")
    fun getAllItems(): Flow<List<ClothingItem>>
    
    @Query("SELECT * FROM clothing_items WHERE id = :id")
    suspend fun getItemById(id: Long): ClothingItem?
    
    @Query("SELECT * FROM clothing_items WHERE type = :type ORDER BY dateAdded DESC")
    fun getItemsByType(type: String): Flow<List<ClothingItem>>
    
    @Query("SELECT * FROM clothing_items WHERE style = :style ORDER BY dateAdded DESC")
    fun getItemsByStyle(style: String): Flow<List<ClothingItem>>
    
    @Query("SELECT * FROM clothing_items WHERE color = :color ORDER BY dateAdded DESC")
    fun getItemsByColor(color: String): Flow<List<ClothingItem>>
    
    @Query("UPDATE clothing_items SET lastWorn = :lastWorn, timesWorn = timesWorn + 1 WHERE id = :id")
    suspend fun updateWearCount(id: Long, lastWorn: LocalDateTime = LocalDateTime.now())
    
    @Query("SELECT * FROM clothing_items ORDER BY timesWorn DESC LIMIT 1")
    suspend fun getMostWornItem(): ClothingItem?
    
    @Query("SELECT * FROM clothing_items WHERE timesWorn > 0 ORDER BY timesWorn ASC LIMIT 1")
    suspend fun getLeastWornItem(): ClothingItem?
}

/**
 * DAO for outfit history
 */
@Dao
interface OutfitHistoryDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(outfitHistory: OutfitHistory): Long
    
    @Update
    suspend fun update(outfitHistory: OutfitHistory)
    
    @Delete
    suspend fun delete(outfitHistory: OutfitHistory)
    
    @Query("SELECT * FROM outfit_history ORDER BY date DESC")
    fun getAllOutfits(): Flow<List<OutfitHistory>>
    
    @Query("SELECT * FROM outfit_history WHERE id = :id")
    suspend fun getOutfitById(id: Long): OutfitHistory?
    
    @Query("SELECT * FROM outfit_history WHERE date BETWEEN :startDate AND :endDate ORDER BY date DESC")
    fun getOutfitsByDateRange(startDate: LocalDateTime, endDate: LocalDateTime): Flow<List<OutfitHistory>>
    
    @Query("SELECT * FROM outfit_history WHERE occasion = :occasion ORDER BY date DESC")
    fun getOutfitsByOccasion(occasion: String): Flow<List<OutfitHistory>>
    
    @Query("SELECT * FROM outfit_history WHERE rating = :rating ORDER BY date DESC")
    fun getOutfitsByRating(rating: Int): Flow<List<OutfitHistory>>
    
    @Query("SELECT AVG(rating) FROM outfit_history WHERE rating IS NOT NULL")
    suspend fun getAverageRating(): Float?
}

/**
 * Repository for wardrobe data
 */
class WardrobeRepository(private val clothingItemDao: ClothingItemDao, private val outfitHistoryDao: OutfitHistoryDao) {
    
    // Clothing item operations
    val allClothingItems = clothingItemDao.getAllItems()
    
    fun getItemsByType(type: String) = clothingItemDao.getItemsByType(type)
    
    fun getItemsByStyle(style: String) = clothingItemDao.getItemsByStyle(style)
    
    fun getItemsByColor(color: String) = clothingItemDao.getItemsByColor(color)
    
    suspend fun getItemById(id: Long) = clothingItemDao.getItemById(id)
    
    suspend fun insertClothingItem(clothingItem: ClothingItem) = clothingItemDao.insert(clothingItem)
    
    suspend fun updateClothingItem(clothingItem: ClothingItem) = clothingItemDao.update(clothingItem)
    
    suspend fun deleteClothingItem(clothingItem: ClothingItem) = clothingItemDao.delete(clothingItem)
    
    suspend fun updateWearCount(id: Long) = clothingItemDao.updateWearCount(id)
    
    suspend fun getMostWornItem() = clothingItemDao.getMostWornItem()
    
    suspend fun getLeastWornItem() = clothingItemDao.getLeastWornItem()
    
    // Outfit history operations
    val allOutfits = outfitHistoryDao.getAllOutfits()
    
    fun getOutfitsByDateRange(startDate: LocalDateTime, endDate: LocalDateTime) = 
        outfitHistoryDao.getOutfitsByDateRange(startDate, endDate)
    
    fun getOutfitsByOccasion(occasion: String) = outfitHistoryDao.getOutfitsByOccasion(occasion)
    
    fun getOutfitsByRating(rating: Int) = outfitHistoryDao.getOutfitsByRating(rating)
    
    suspend fun getOutfitById(id: Long) = outfitHistoryDao.getOutfitById(id)
    
    suspend fun insertOutfit(outfitHistory: OutfitHistory) = outfitHistoryDao.insert(outfitHistory)
    
    suspend fun updateOutfit(outfitHistory: OutfitHistory) = outfitHistoryDao.update(outfitHistory)
    
    suspend fun deleteOutfit(outfitHistory: OutfitHistory) = outfitHistoryDao.delete(outfitHistory)
    
    suspend fun getAverageRating() = outfitHistoryDao.getAverageRating()
    
    /**
     * Generate outfit recommendations based on occasion, weather, and user preferences
     */
    suspend fun generateOutfitRecommendations(
        occasion: String,
        temperature: Double? = null,
        weatherCondition: String? = null,
        preferredStyles: List<String> = listOf(),
        preferredColors: List<String> = listOf()
    ): List<OutfitRecommendation> {
        // Get all clothing items
        val allItems = allClothingItems.value ?: return emptyList()
        
        // Filter tops
        val tops = allItems.filter { it.type == "Top" || it.type == "Shirt" || it.type == "T-shirt" }
        
        // Filter bottoms
        val bottoms = allItems.filter { it.type == "Bottom" || it.type == "Pants" || it.type == "Jeans" || it.type == "Skirt" }
        
        // Filter shoes
        val shoes = allItems.filter { it.type == "Shoes" || it.type == "Footwear" }
        
        // Filter outerwear
        val outerwear = allItems.filter { it.type == "Outerwear" || it.type == "Jacket" || it.type == "Coat" }
        
        // Filter by occasion
        val occasionTops = filterByOccasion(tops, occasion)
        val occasionBottoms = filterByOccasion(bottoms, occasion)
        val occasionShoes = filterByOccasion(shoes, occasion)
        val occasionOuterwear = filterByOccasion(outerwear, occasion)
        
        // Filter by weather if available
        val weatherTops = if (temperature != null) filterByWeather(occasionTops, temperature, weatherCondition) else occasionTops
        val weatherBottoms = if (temperature != null) filterByWeather(occasionBottoms, temperature, weatherCondition) else occasionBottoms
        val weatherShoes = if (temperature != null) filterByWeather(occasionShoes, temperature, weatherCondition) else occasionShoes
        val weatherOuterwear = if (temperature != null) filterByWeather(occasionOuterwear, temperature, weatherCondition) else occasionOuterwear
        
        // Filter by preferred styles if available
        val styleTops = if (preferredStyles.isNotEmpty()) filterByStyles(weatherTops, preferredStyles) else weatherTops
        val styleBottoms = if (preferredStyles.isNotEmpty()) filterByStyles(weatherBottoms, preferredStyles) else weatherBottoms
        val styleShoes = if (preferredStyles.isNotEmpty()) filterByStyles(weatherShoes, preferredStyles) else weatherShoes
        val styleOuterwear = if (preferredStyles.isNotEmpty()) filterByStyles(weatherOuterwear, preferredStyles) else weatherOuterwear
        
        // Filter by preferred colors if available
        val colorTops = if (preferredColors.isNotEmpty()) filterByColors(styleTops, preferredColors) else styleTops
        val colorBottoms = if (preferredColors.isNotEmpty()) filterByColors(styleBottoms, preferredColors) else styleBottoms
        val colorShoes = if (preferredColors.isNotEmpty()) filterByColors(styleShoes, preferredColors) else styleShoes
        val colorOuterwear = if (preferredColors.isNotEmpty()) filterByColors(styleOuterwear, preferredColors) else styleOuterwear
        
        // Generate outfit combinations
        val recommendations = mutableListOf<OutfitRecommendation>()
        
        // Limit to 5 recommendations
        val maxRecommendations = 5
        
        // Generate combinations
        for (top in colorTops.take(3)) {
            for (bottom in colorBottoms.take(3)) {
                for (shoe in colorShoes.take(2)) {
                    // Check if we have enough recommendations
                    if (recommendations.size >= maxRecommendations) break
                    
                    // Add outerwear if available and appropriate for weather
                    val outerwearItem = if (colorOuterwear.isNotEmpty() && temperature != null && temperature < 20) {
                        colorOuterwear.firstOrNull()
                    } else {
                        null
                    }
                    
                    // Create recommendation
                    val recommendation = OutfitRecommendation(
                        name = "${top.style} Outfit",
                        top = top,
                        bottom = bottom,
                        shoes = shoe,
                        outerwear = outerwearItem,
                        occasion = occasion,
                        weatherCompatible = isWeatherCompatible(top, bottom, shoe, outerwearItem, temperature, weatherCondition)
                    )
                    
                    recommendations.add(recommendation)
                }
                
                // Check if we have enough recommendations
                if (recommendations.size >= maxRecommendations) break
            }
            
            // Check if we have enough recommendations
            if (recommendations.size >= maxRecommendations) break
        }
        
        return recommendations
    }
    
    /**
     * Filter clothing items by occasion
     */
    private fun filterByOccasion(items: List<ClothingItem>, occasion: String): List<ClothingItem> {
        return when (occasion.lowercase()) {
            "formal" -> items.filter { it.style.lowercase() in listOf("formal", "business", "elegant") }
            "business" -> items.filter { it.style.lowercase() in listOf("business", "formal", "professional") }
            "casual" -> items.filter { it.style.lowercase() in listOf("casual", "everyday", "relaxed") }
            "sport" -> items.filter { it.style.lowercase() in listOf("sport", "athletic", "active") }
            else -> items
        }
    }
    
    /**
     * Filter clothing items by weather
     */
    private fun filterByWeather(items: List<ClothingItem>, temperature: Double, weatherCondition: String?): List<ClothingItem> {
        return items.filter { item ->
            when {
                // Hot weather
                temperature > 30 -> {
                    when (item.type.lowercase()) {
                        "top", "shirt", "t-shirt" -> item.material.lowercase() in listOf("cotton", "linen")
                        "bottom", "pants", "jeans", "skirt" -> item.material.lowercase() in listOf("cotton", "linen") || item.type.lowercase() == "shorts"
                        "outerwear", "jacket", "coat" -> false // No outerwear in hot weather
                        else -> true
                    }
                }
                // Warm weather
                temperature in 20.0..30.0 -> {
                    when (item.type.lowercase()) {
                        "outerwear", "jacket", "coat" -> item.material.lowercase() in listOf("cotton", "linen")
                        else -> true
                    }
                }
                // Cool weather
                temperature in 10.0..20.0 -> {
                    when (item.type.lowercase()) {
                        "top", "shirt", "t-shirt" -> true
                        "bottom", "pants", "jeans", "skirt" -> item.type.lowercase() != "shorts"
                        else -> true
                    }
                }
                // Cold weather
                temperature < 10 -> {
                    when (item.type.lowercase()) {
                        "top", "shirt", "t-shirt" -> item.material.lowercase() in listOf("wool", "fleece", "cotton")
                        "bottom", "pants", "jeans", "skirt" -> item.type.lowercase() != "shorts" && item.material.lowercase() != "linen"
                        "outerwear", "jacket", "coat" -> true
                        else -> true
                    }
                }
                else -> true
            }
        }
    }
    
    /**
     * Filter clothing items by styles
     */
    private fun filterByStyles(items: List<ClothingItem>, styles: List<String>): List<ClothingItem> {
        val lowerCaseStyles = styles.map { it.lowercase() }
        return items.filter { item -> item.style.lowercase() in lowerCaseStyles }
    }
    
    /**
     * Filter clothing items by colors
     */
    private fun filterByColors(items: List<ClothingItem>, colors: List<String>): List<ClothingItem> {
        val lowerCaseColors = colors.map { it.lowercase() }
        return items.filter { item -> item.color.lowercase() in lowerCaseColors }
    }
    

(Content truncated due to size limit. Use line ranges to read in chunks)