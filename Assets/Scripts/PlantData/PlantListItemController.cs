using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO;

public class PlantListItemController : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI nameText;
    [SerializeField] private TextMeshProUGUI timeText;
    [SerializeField] private RawImage thumbnailImage; // optional; can be null if unused

    /// <summary>
    /// Initialize this UI item from a PlantEntry.
    /// If entry.thumbnailFileName is provided, attempts to load from persistentDataPath.
    /// </summary>
    public void Initialize(PlantEntry entry)
    {
        if (nameText != null)
            nameText.text = entry.plantName;

        if (timeText != null)
        {
            // Parse or display raw timestamp. Here we show local time:
            if (System.DateTime.TryParse(entry.timestamp, null, System.Globalization.DateTimeStyles.RoundtripKind, out var dt))
            {
                // Display in local time or custom format
                timeText.text = dt.ToLocalTime().ToString("yyyy-MM-dd HH:mm");
            }
            else
            {
                timeText.text = entry.timestamp;
            }
        }

        if (thumbnailImage != null)
        {
            if (!string.IsNullOrEmpty(entry.thumbnailFileName))
            {
                string path = Path.Combine(Application.persistentDataPath, entry.thumbnailFileName);
                if (File.Exists(path))
                {
                    byte[] bytes = File.ReadAllBytes(path);
                    Texture2D tex = new Texture2D(2, 2);
                    if (tex.LoadImage(bytes))
                    {
                        thumbnailImage.texture = tex;
                        // Optionally set size, aspect, etc.
                    }
                }
                else
                {
                    // No file: optionally hide thumbnailImage.gameObject
                    thumbnailImage.gameObject.SetActive(false);
                }
            }
            else
            {
                thumbnailImage.gameObject.SetActive(false);
            }
        }
    }
}
