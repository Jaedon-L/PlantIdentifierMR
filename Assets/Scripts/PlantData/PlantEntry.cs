using System;

[Serializable]
public class PlantEntry
{
    public string plantName;
    public string timestamp;    // e.g. DateTime.UtcNow.ToString("o")
    public string thumbnailFileName; // optional: e.g. "plant_2025-06-12T15-30-00.png"

    public PlantEntry(string name, string ts, string thumbFileName = null)
    {
        plantName = name;
        timestamp = ts;
        thumbnailFileName = thumbFileName;
    }
}

/// JsonUtility cannot directly serialize List<T> at root, so wrap it:
[Serializable]
public class PlantCollectionData
{
    public PlantEntry[] entries;
}
