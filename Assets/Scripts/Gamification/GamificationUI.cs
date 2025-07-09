using TMPro;
using UnityEngine;

public class GamificationUI : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI levelText;
    [SerializeField] private TextMeshProUGUI xpText;
    [SerializeField] private TextMeshProUGUI achievementText;  // ➕ Add this in the Inspector

    private void Update()
    {
        if (GamificationManager.Instance == null) return;

        // Show Level & XP
        levelText.text = $"Level {GamificationManager.Instance.Level}";
        int currentXP = GamificationManager.Instance.XP;
        int nextLevelXP = GamificationManager.Instance.GetNextLevelThreshold();
        xpText.text = $"XP: {currentXP} / {nextLevelXP}";

        // Show achievements
        achievementText.text = "";
        foreach (var name in GamificationManager.Instance.GetAllAchievementNames())
        {
            bool unlocked = GamificationManager.Instance.IsAchievementUnlocked(name);
            achievementText.text += unlocked ? $"✅ {name}\n" : $"❌ {name}\n";
        }
    }
}