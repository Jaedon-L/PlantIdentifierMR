using UnityEngine;
using LLMUnity;

[RequireComponent(typeof(LLMCharacter))]
public class GrammarInjector : MonoBehaviour
{
    // Path under Resources/, omit extension
    const string k_GrammarResourcePath = "grammar/json";

    void Awake()
    {
        // 1) Grab the LLMCharacter
        var llm = GetComponent<LLMCharacter>();

        // 2) Load the grammar from Resources
        var asset = Resources.Load<TextAsset>(k_GrammarResourcePath);
        if (asset == null)
        {
            Debug.LogError($"[GrammarInjector] Failed to load Resources/{k_GrammarResourcePath}.txt as a TextAsset");
            return;
        }

        // 3) Overwrite whatever InitGrammar would do
        llm.grammar = "";           // clear any inspector path
        llm.grammarJSON = "";       // clear JSON-grammar slot
        llm.grammarString = asset.text;
        llm.grammarJSONString = "";

        Debug.Log("[GrammarInjector] Injected grammar into LLMCharacter.");
    }
}