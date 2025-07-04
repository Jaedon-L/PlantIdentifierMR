using TMPro;
using UnityEngine;

public class GamificationUI : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI levelText;
    [SerializeField] private TextMeshProUGUI xpText;

    private void Update()
    {
        if (GamificationManager.Instance == null) return;

        levelText.text = $"Level {GamificationManager.Instance.Level}";
        int currentXP = GamificationManager.Instance.XP;
        int nextLevelXP = GamificationManager.Instance.GetNextLevelThreshold();
        xpText.text = $"XP: {currentXP} / {nextLevelXP}";
    }
}
