using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Gamification Manager: Handles XP, level, and basic achievements.
/// Supports adding plants and watering. Expandable for future features.
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
    [SerializeField] private int xpPerWater = 5;

    [Header("Achievements")]
    private HashSet<string> unlockedAchievements = new HashSet<string>();

    private int waterCount = 0;
    private int loginCount = 0; // ➕ NEW: Count how many times app opened

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

        OnAppOpened(); // ➕ Call app open logic on start
    }

    public void OnAppOpened()
    {
        AddXP(1); // ➕ Give XP every open
        loginCount++;

        if (loginCount == 1)
            TryUnlockAchievement("First Step");
        if (loginCount == 15)
            TryUnlockAchievement("Plant Lover");

        SaveProgress();
        NotifyUI();
    }

    public void OnPlantAdded(string plantName)
    {
        AddXP(xpPerPlantAdded);
        TryUnlockAchievement("Green Thumb");
        SaveProgress();
        NotifyUI();
    }

    public void OnWatered()
    {
        AddXP(xpPerWater);
        waterCount++;

        if (waterCount == 1)
            TryUnlockAchievement("Water Novice");
        if (waterCount == 5)
            TryUnlockAchievement("Hydration Hero");

        SaveProgress();
        NotifyUI();
    }

    public void AddXP(int amount)
    {
        xp += amount;
        Debug.Log($"Gained {amount} XP. Total XP: {xp}");

        int newLevel = CalculateLevel();
        if (newLevel > level)
        {
            level = newLevel;
            Debug.Log($"Level up! New level: {level}");
        }

        NotifyUI();
        SaveProgress();
    }

    private int CalculateLevel()
    {
        for (int i = levelThresholds.Count - 1; i >= 0; i--)
        {
            if (xp >= levelThresholds[i])
                return i + 1;
        }
        return 1;
    }

    private void TryUnlockAchievement(string achievementName)
    {
        if (!unlockedAchievements.Contains(achievementName))
        {
            unlockedAchievements.Add(achievementName);
            Debug.Log($"Achievement Unlocked: {achievementName}");

            NotifyUI();
            SaveProgress();
        }
    }

    public List<string> GetAllAchievementNames()
    {
        return new List<string> { "Green Thumb", "Water Novice", "Hydration Hero", "First Step", "Plant Lover" }; // ➕ Add new achievement names
    }

    public bool IsAchievementUnlocked(string name)
    {
        return unlockedAchievements.Contains(name);
    }

    public int GetNextLevelThreshold()
    {
        int nextLevelIndex = level;
        if (nextLevelIndex >= levelThresholds.Count)
            return levelThresholds[levelThresholds.Count - 1];

        return levelThresholds[nextLevelIndex];
    }

    private void NotifyUI()
    {
        var ui = FindObjectOfType<GamificationUI>();
        if (ui != null)
        {
            ui.RefreshUI();
        }
    }

    private void SaveProgress()
    {
        GamificationData data = new GamificationData
        {
            xp = this.xp,
            level = this.level,
            achievements = new List<string>(unlockedAchievements),
            loginCount = this.loginCount // ➕ Save login count
        };
        string json = JsonUtility.ToJson(data, prettyPrint: true);
        File.WriteAllText(saveFilePath, json);
        Debug.Log("GamificationManager: Progress saved.");
    }

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
            loginCount = data.loginCount; // ➕ Load login count
            Debug.Log("GamificationManager: Progress loaded.");
        }
        catch (Exception e)
        {
            Debug.LogError($"GamificationManager: Failed to load progress: {e}");
        }
    }
}

[Serializable]
public class GamificationData
{
    public int xp;
    public int level;
    public List<string> achievements;
    public int loginCount; // ➕ Add to save the login count
}
