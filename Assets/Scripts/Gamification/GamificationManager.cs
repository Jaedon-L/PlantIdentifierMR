using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Simple Gamification Manager: Handles XP, level, and basic achievements.
/// Expandable for future actions (harvest, water, etc.).
/// </summary>
public class GamificationManager : MonoBehaviour
{
    public static GamificationManager Instance { get; private set; }

    [Header("Level Settings")]
    [SerializeField] private int xp = 0;
    [SerializeField] private int level = 1;

    [Tooltip("XP required to reach each level. Configure in the Inspector.")]
    [SerializeField] private List<int> levelThresholds = new List<int> { 0, 50, 120, 210, 320, 450, 600, 770, 960, 1170 };

    [Header("XP Rewards for Actions")]
    [SerializeField] private int xpPerPlantAdded = 10;
    [SerializeField] private int xpPerHarvest = 15;
    [SerializeField] private int xpPerWater = 5;

    [Header("Achievements")]
    private HashSet<string> unlockedAchievements = new HashSet<string>();

    private string saveFilePath;

    public int XP => xp;
    public int Level => level;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);

        saveFilePath = Path.Combine(Application.persistentDataPath, "gamification_data.json");
        LoadProgress();
    }

    /// <summary>
    /// Call this when a new plant is added.
    /// </summary>
    public void OnPlantAdded(string plantName)
    {
        AddXP(xpPerPlantAdded);
        TryUnlockAchievement("Green Thumb"); // Example achievement
    }

    /// <summary>
    /// Add XP and check for level up.
    /// </summary>
    public void AddXP(int amount)
    {
        xp += amount;
        Debug.Log($"Gained {amount} XP. Total XP: {xp}");

        int newLevel = CalculateLevel();
        if (newLevel > level)
        {
            level = newLevel;
            Debug.Log($"Level up! New level: {level}");
            // TODO: Trigger UI feedback (animation, sound, etc.)
        }

        SaveProgress();
    }

    /// <summary>
    /// Determines the level based on XP.
    /// </summary>
    private int CalculateLevel()
    {
        for (int i = levelThresholds.Count - 1; i >= 0; i--)
        {
            if (xp >= levelThresholds[i])
                return i + 1;
        }
        return 1;
    }

    /// <summary>
    /// Try to unlock an achievement by name.
    /// </summary>
    private void TryUnlockAchievement(string achievementName)
    {
        if (!unlockedAchievements.Contains(achievementName))
        {
            unlockedAchievements.Add(achievementName);
            Debug.Log($"Achievement Unlocked: {achievementName}");
            // TODO: Trigger UI popup, badge, etc.
            SaveProgress();
        }
    }

    /// <summary>
    /// Save XP, level, and achievements to disk.
    /// </summary>
    private void SaveProgress()
    {
        GamificationData data = new GamificationData
        {
            xp = this.xp,
            level = this.level,
            achievements = new List<string>(unlockedAchievements)
        };
        string json = JsonUtility.ToJson(data, prettyPrint: true);
        File.WriteAllText(saveFilePath, json);
        Debug.Log("GamificationManager: Progress saved.");
    }

    /// <summary>
    /// Load XP, level, and achievements from disk.
    /// </summary>
    private void LoadProgress()
    {
        if (!File.Exists(saveFilePath))
        {
            Debug.Log("GamificationManager: No save file found. Starting fresh.");
            return;
        }

        try
        {
            string json = File.ReadAllText(saveFilePath);
            GamificationData data = JsonUtility.FromJson<GamificationData>(json);
            xp = data.xp;
            level = data.level;
            unlockedAchievements = new HashSet<string>(data.achievements);
            Debug.Log("GamificationManager: Progress loaded.");
        }
        catch (Exception e)
        {
            Debug.LogError($"GamificationManager: Failed to load progress: {e}");
        }
    }

    public int GetNextLevelThreshold()
    {
        if (level - 1 < levelThresholds.Count)
            return levelThresholds[level - 1];
        return levelThresholds[levelThresholds.Count - 1]; // return last threshold if maxed
    }
}

/// <summary>
/// Serializable wrapper for gamification data.
/// </summary>
[Serializable]
public class GamificationData
{
    public int xp;
    public int level;
    public List<string> achievements;
}
