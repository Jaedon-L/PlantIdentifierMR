using System;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using LLMUnity;
using System.Text.RegularExpressions;
using System.IO;
using System.Linq;

[Serializable]
public class CareGuide
{
    public string watering;
    public string light;
    // public string soil;
    public string fertilizer;
}

public class PlantAssistant1 : MonoBehaviour
{
    [Header("UI References")]
    public TMP_InputField plantInput;
    public TMP_Text outputText;

    [Header("LLMUnity Integration")]
    public LLMCharacter llmCharacter;

    async void Start()
    {
        // Warm up once at startup to avoid the first-call delay
        await llmCharacter.Warmup();
    }


    private void HandleLLMReply(string reply, string plantName)
    {
        if (!reply.Contains("}")) return;
        Debug.Log($"[Raw LLM Reply] {reply}");
        string json = ExtractAndFixJson(reply);
        if (string.IsNullOrEmpty(json))
        {
            outputText.text = "Error: Could not extract care info.";
            return;
        }

        CareGuide guide;
        try
        {
            guide = JsonUtility.FromJson<CareGuide>(json);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[ParseError] JSON: {json}\n{ex}");
            outputText.text = "Error parsing care guide.";
            return;
        }

        // If a field is still empty, give it a fallback label
        if (string.IsNullOrEmpty(guide.watering)) guide.watering = "N/A";
        if (string.IsNullOrEmpty(guide.light)) guide.light = "N/A";
        if (string.IsNullOrEmpty(guide.fertilizer)) guide.fertilizer = "N/A";

        outputText.text =
            $"• Plant: {plantName}\n" +
            $"• Watering: {guide.watering}\n" +
            $"• Light:    {guide.light}\n" +
            $"• Fertilizer: {guide.fertilizer}";

        PlantDatabase.SaveNewPlant(plantName, guide);
    }


    [ContextMenu("button press")]
    public void OnAskButtonPressed()
    {
        llmCharacter.ClearChat();
        // 1) Make sure our JSON DB is ready
        if (PlantDatabase.Plants == null)
        {
            outputText.text = "Loading plant data… please wait.";
            return;
        }

        // 2) Grab & validate the plant name
        string plantName = plantInput.text?.Trim();
        if (string.IsNullOrEmpty(plantName))
        {
            outputText.text = "Please enter a plant name.";
            return;
        }

        // 3) If we have curated data, use it
        // if (PlantDatabase.Plants.TryGetValue(plantName, out var info))
        // {
        // Build a succinct prompt (system prompt steers to bullets only)
        // string prompt = $@"

        // {info.watering}
        // {info.light}
        // {info.soil}
        // {info.fertilizer}
        // ";
        //             string prompt = $@"
        // You are a concise plant-care assistant.
        // Return this plant care data in JSON:
        // {{
        //   ""Watering"": ""{info.watering}"",
        //   ""Light"": ""{info.light}"",
        //   ""Fertilizer"": ""{info.fertilizer}""
        // }}
        // ";

        //             outputText.text = "Thinking…";

        //             // Fire off the chat; reply callback updates the UI
        //             _ = llmCharacter.Chat(
        //                 prompt,
        //                 reply => outputText.text = reply
        //             );
        // Check existing DB
        if (PlantDatabase.Plants.TryGetValue(plantName, out var info))
        {
            outputText.text =
                $"• Plant: {plantName}\n" +
                $"• Watering: {info.watering}\n" +
                $"• Light: {info.light}\n" +
                $"• Fertilizer: {info.fertilizer}";
            return;

        }
        else
        {
            outputText.text = "Looking up care instructions…";

            // Fallback with a single reference to the plant name
            //             string fallbackPrompt = $@"
            // Plant: {plantName}

            // No DB data—please provide four care bullets:
            // • watering
            // • light
            // • soil
            // • fertilizer
            // ";
            // string fallbackPrompt = $@"
            // You are a concise plant-care assistant.

            // Only respond in this exact JSON format with 3 fields :
            // {{
            //   ""Watering"": ""<x-y times per week only>"",
            //   ""Light"": ""<x-y hours sunlight per day only>"",
            //   ""Fertilizer"": ""<x-y times per year only>""
            // }}

            // Plant: {plantName}
            // ";
            string fallbackPrompt = BuildResearchPrompt(plantName);
            _ = llmCharacter.Complete(
                fallbackPrompt,
                // reply => outputText.text = reply,
                reply => HandleLLMReply(reply, plantName),
                () => Debug.Log($"[PlantAssistant] Fallback for {plantName} complete")
            );
        }
    }

    // Optional: allow cancelling the current request (e.g. a Stop button)
    public void OnCancelButtonPressed()
    {
        llmCharacter.CancelRequests();
        outputText.text = "Request cancelled.";
    }
    private string ExtractAndFixJson(string raw)
    {
        int start = raw.IndexOf('{');
        int end = raw.LastIndexOf('}');
        if (start < 0 || end <= start) return null;

        string json = raw.Substring(start, end - start + 1);

        // Step 1: original regex corrections
        json = Regex.Replace(json, "\"\\s*Watering\\s*\"\\s*:", "\"watering\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Light\\s*\"\\s*:", "\"light\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Lihr\\s*\"\\s*:", "\"light\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Liigh\\s*\"\\s*:", "\"light\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Liight\\s*\"\\s*:", "\"light\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Lihrt\\s*\"\\s*:", "\"light\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*LiHit\\s*\"\\s*:", "\"light\":", RegexOptions.IgnoreCase);

        json = Regex.Replace(json, "\"\\s*Fertilizer\\s*\"\\s*:", "\"fertilizer\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Fehr\\s*\"\\s*:", "\"fertilizer\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Fehrtilizer\\s*\"\\s*:", "\"fertilizer\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Feirtilizer\\s*\"\\s*:", "\"fertilizer\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*FeRtIl\\s*\"\\s*:", "\"fertilizer\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Feert\\s*\"\\s*:", "\"fertilizer\":", RegexOptions.IgnoreCase);
        json = Regex.Replace(json, "\"\\s*Fehrtil\\s*\"\\s*:", "\"fertilizer\":", RegexOptions.IgnoreCase);

        // Step 2: fuzzy fallback extraction
        var matches = Regex.Matches(json, "\"\\s*([^\"]+?)\\s*\"\\s*:\\s*\"([^\"]*?)\"");
        string watering = null, light = null, fertilizer = null;
        foreach (Match m in matches)
        {
            string keyRaw = m.Groups[1].Value.Replace(" ", "").ToLowerInvariant();
            string value = m.Groups[2].Value;

            if (keyRaw.Contains("water")) watering = value;
            else if (keyRaw.Contains("light") || keyRaw.Contains("ligh") || keyRaw.Contains("lihgt") || keyRaw.Contains("lihg")||keyRaw.Contains("lih") ) light = value;
            else if (keyRaw.Contains("fert") || keyRaw.Contains("fer") || keyRaw.Contains("fe")) fertilizer = value;
        }

        var clean = "{";
        if (!string.IsNullOrEmpty(watering)) clean += $"\"watering\":\"{watering}\",";
        if (!string.IsNullOrEmpty(light)) clean += $"\"light\":\"{light}\",";
        if (!string.IsNullOrEmpty(fertilizer)) clean += $"\"fertilizer\":\"{fertilizer}\",";
        if (clean.EndsWith(",")) clean = clean.Substring(0, clean.Length - 1);
        clean += "}";

        Debug.Log($"[Extracted JSON] {clean}");
        return clean;
    }

    string BuildResearchPrompt(string plantName)
    {
        // Pull 2 random examples from your loaded DB
        var examples = PlantDatabase.Plants
            .OrderBy(_ => Guid.NewGuid())
            .Take(3)
            .Select(kv => $@"
Plant: {kv.Key}
{{
  ""Watering"": ""{kv.Value.watering}"",
  ""Light"":     ""{kv.Value.light}"",
  ""Fertilizer"":""{kv.Value.fertilizer}""
}}")
            .Aggregate("", (a, b) => a + b);

        // Now append the “new” plant you want researched
        return $@"
You are a world‑class botanist with encyclopedic knowledge of plant care.

Here are examples of real researched data:
{examples}

Now, for the new plant below, *research* and respond *only* with this exact JSON format—no bullets, no extra text, and **exactly these three keys** (in this order):
IMPORTANT: Respond with exactly ONE JSON object, no extra text, and exactly three keys:
- ""Watering""
- ""Light""
- ""Fertilizer""

Do NOT rename or misspell the keys. Do NOT add extra text. Reply ONLY with the JSON:
{{
  ""Watering"": ""<x–y times per week>"",
  ""Light"":     ""<x–y **hours** of sunlight per day>"",
  ""Fertilizer"":""<x–y times per year>""
}}

Replace the placeholders with *actual values* for the given plant.


Plant: {plantName}
";
    }


}
// [ContextMenu("button press")]
// public void OnAskButtonPressed()
// {
//     llmCharacter.ClearChat();

//     string plantName = plantInput.text?.Trim();
//     if (string.IsNullOrEmpty(plantName))
//     {
//         outputText.text = "Please enter a plant name.";
//         return;
//     }

//     // Check existing DB
//     if (PlantDatabase.Plants.TryGetValue(plantName, out var existingInfo))
//     {
//         outputText.text =
//             $"• Watering: {existingInfo.watering}\n" +
//             $"• Light: {existingInfo.light}\n" +
//             $"• Fertilizer: {existingInfo.fertilizer}";
//         return;
//     }

//     string prompt = $@"
//     You are a concise plant-care assistant.

//     Only respond in this exact JSON format with 3 fields and no extra text:
//     {{
//       ""Watering"": ""<x-y times per week only>"",
//       ""Light"": ""<x-y hours sunlight per day only>"",
//       ""Fertilizer"": ""<x-y times per year only>""
//     }}

//     Plant: {plantName}
//     ";

//     outputText.text = "Thinking…";

//     llmCharacter.Complete(
//         prompt,
//         reply => HandleLLMReply(reply, plantName)
//     );
// }