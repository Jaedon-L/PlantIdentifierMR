using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;  // for UnityWebRequest

[Serializable]
public class PlantCareInfo {
    public string description;
    public string watering;
    public string light;
    public string soil;
    public string fertilizer;
}

[Serializable]
public class PlantCareEntry {
    public string name;
    public PlantCareInfo info;
}

[Serializable]
public class PlantCareDb {
    public List<PlantCareEntry> entries;
}

// public class PlantDatabase : MonoBehaviour {
//     // Use a case-insensitive dictionary for lookups:
//     public static Dictionary<string, PlantCareInfo> Plants;

//     private void Awake() {
//         StartCoroutine(LoadPlantDatabase());
//     }

//     private System.Collections.IEnumerator LoadPlantDatabase() {
//         string fileName = "data/plant_db.json";
//         string fullPath = Path.Combine(Application.streamingAssetsPath, fileName);
//         string json;

//         if (fullPath.Contains("://") || fullPath.Contains(":///")) {
//             // Android / WebGL
//             using var www = UnityWebRequest.Get(fullPath);
//             yield return www.SendWebRequest();
//             if (www.result != UnityWebRequest.Result.Success) {
//                 Debug.LogError($"[PlantDatabase] Failed to load JSON: {www.error}");
//                 yield break;
//             }
//             json = www.downloadHandler.text;
//         } else {
//             // Editor / Standalone
//             if (!File.Exists(fullPath)) {
//                 Debug.LogError($"[PlantDatabase] JSON not found at {fullPath}");
//                 yield break;
//             }
//             json = File.ReadAllText(fullPath);
//         }

//         var wrapper = JsonUtility.FromJson<PlantCareDb>(json);
//         if (wrapper?.entries == null) {
//             Debug.LogError("[PlantDatabase] Failed to parse JSON into PlantCareDb.");
//             yield break;
//         }

//         // Initialize with case-insensitive keys
//         Plants = new Dictionary<string, PlantCareInfo>(StringComparer.OrdinalIgnoreCase);

//         foreach (var e in wrapper.entries) {
//             if (string.IsNullOrWhiteSpace(e.name)) {
//                 Debug.LogWarning("[PlantDatabase] Skipping entry with empty name");
//                 continue;
//             }
//             Plants[e.name] = e.info;
//         }

//         Debug.Log($"[PlantDatabase] Loaded {Plants.Count} plant entries.");
//         // Debug-log all keys so you know exactly what names are available:
//         foreach (var key in Plants.Keys) {
//             Debug.Log($"[PlantDatabase] Key: '{key}'");
//         }
//     }
// }
public class PlantDatabase : MonoBehaviour
{
    public static Dictionary<string, PlantCareInfo> Plants;

    private string builtInPath => Path.Combine(Application.streamingAssetsPath, "data/plant_db.json");
    private string userDbPath => Path.Combine(Application.persistentDataPath, "user_plant_db.json");

    private void Awake()
    {
        Plants = new Dictionary<string, PlantCareInfo>(StringComparer.OrdinalIgnoreCase);
        StartCoroutine(LoadDatabases());
    }

    public IEnumerator LoadDatabases()
    {
        // 1. Load built-in database
        yield return LoadDatabaseFromPath(builtInPath);

        // 2. Load user-generated database
        if (File.Exists(userDbPath))
        {
            string json = File.ReadAllText(userDbPath);
            var userDb = JsonUtility.FromJson<PlantCareDb>(json);
            if (userDb?.entries != null)
            {
                foreach (var entry in userDb.entries)
                {
                    if (!string.IsNullOrWhiteSpace(entry.name))
                        Plants[entry.name] = entry.info; // overwrite if duplicate
                }
            }
        }

        Debug.Log($"[PlantDatabase] Final merged entries: {Plants.Count}");
    }

    private IEnumerator LoadDatabaseFromPath(string path)
    {
        string json;
        if (path.Contains("://") || path.Contains(":///"))
        {
            using var www = UnityWebRequest.Get(path);
            yield return www.SendWebRequest();
            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[PlantDatabase] Failed to load JSON from {path}: {www.error}");
                yield break;
            }
            json = www.downloadHandler.text;
        }
        else
        {
            if (!File.Exists(path))
            {
                Debug.LogWarning($"[PlantDatabase] File not found: {path}");
                yield break;
            }
            json = File.ReadAllText(path);
        }

        var db = JsonUtility.FromJson<PlantCareDb>(json);
        if (db?.entries != null)
        {
            foreach (var entry in db.entries)
            {
                if (!string.IsNullOrWhiteSpace(entry.name))
                    Plants[entry.name] = entry.info;
            }
        }
    }

    // Call this from PlantAssistant to update the user DB
    public static void SaveNewPlant(string plantName, CareGuide guide)
    {
        var userDb = new PlantCareDb { entries = new List<PlantCareEntry>() };

        // Load existing if it exists
        string path = Path.Combine(Application.persistentDataPath, "user_plant_db.json");
        if (File.Exists(path))
        {
            var existing = JsonUtility.FromJson<PlantCareDb>(File.ReadAllText(path));
            if (existing?.entries != null)
                userDb.entries = existing.entries;
        }

        // Replace or Add
        var existingEntry = userDb.entries.Find(e => e.name.Equals(plantName, StringComparison.OrdinalIgnoreCase));
        if (existingEntry != null)
            existingEntry.info = new PlantCareInfo { watering = guide.watering, light = guide.light, fertilizer = guide.fertilizer };
        else
            userDb.entries.Add(new PlantCareEntry
            {
                name = plantName,
                info = new PlantCareInfo { watering = guide.watering, light = guide.light, fertilizer = guide.fertilizer }
            });

        File.WriteAllText(path, JsonUtility.ToJson(userDb, true));
        Plants[plantName] = new PlantCareInfo { watering = guide.watering, light = guide.light, fertilizer = guide.fertilizer };
        Debug.Log($"[PlantDatabase] Saved new entry: {plantName}");
    }
}
