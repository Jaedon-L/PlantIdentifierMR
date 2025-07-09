using TMPro;
using UnityEngine;

public class GamificationUI : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI levelText;
    [SerializeField] private TextMeshProUGUI xpText;
    [SerializeField] private TextMeshProUGUI unlockedText;
    [SerializeField] private TextMeshProUGUI lockedText;
    [SerializeField] private GameObject targetPanel;

    private void Start()
    {
        RefreshUI();  // Initial refresh when scene loads
    }

    public void RefreshUI()
    {
        if (GamificationManager.Instance == null) return;

        // Show Level & XP
        levelText.text = $"Level {GamificationManager.Instance.Level}";
        int currentXP = GamificationManager.Instance.XP;
        int nextLevelXP = GamificationManager.Instance.GetNextLevelThreshold();
        xpText.text = $"XP: {currentXP} / {nextLevelXP}";

        // Separate achievements
        unlockedText.text = "✅ Completed:\n";
        lockedText.text = "❌ Not yet:\n";

        foreach (var name in GamificationManager.Instance.GetAllAchievementNames())
        {
            if (GamificationManager.Instance.IsAchievementUnlocked(name))
            {
                unlockedText.text += $"• {name}\n";
            }
            else
            {
                lockedText.text += $"• {name}\n";
            }
        }

        if (unlockedText.text == "✅ Completed:\n")
            unlockedText.text += "None yet.";

        if (lockedText.text == "❌ Not yet:\n")
            lockedText.text += "All done!";
    }

    public void ToggleVisibility()
    {
        if (targetPanel != null)
        {
            bool isActive = targetPanel.activeSelf;
            targetPanel.SetActive(!isActive);

            if (targetPanel.activeSelf)
            {
                RefreshUI();
            }
        }
    }
}