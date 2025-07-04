using UnityEngine;
using TMPro;

public class WateringHistoryItem : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI historyText;

    public void SetText(string value)
    {
        if (historyText != null)
            historyText.text = value;
    }
}
