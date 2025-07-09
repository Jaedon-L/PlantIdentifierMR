﻿using System;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class WateringReminder : MonoBehaviour
{
    [SerializeField] private TMP_InputField daysInput;
    [SerializeField] private TMP_InputField hoursInput;
    [SerializeField] private TMP_InputField minutesInput;
    [SerializeField] private Button setButton;

    [SerializeField] private GameObject reminderUI; 
    [SerializeField] private AudioSource alertAudio; 
    [SerializeField] private Slider wateringSlider; 
    [SerializeField] private TextMeshProUGUI countdownText; 
    //new
    [SerializeField] private Button wateredButton; 
   // [SerializeField] private TextMeshProUGUI historyText; 
    //new
    [SerializeField] private GameObject historyItemPrefab; 
    [SerializeField] private Transform historyPanel; 

    //new
    [SerializeField] private Button togglePanelButton; 
    [SerializeField] private GameObject targetPanel;   
    private bool panelVisible = false; 
    // private System.Text.StringBuilder wateringHistory = new System.Text.StringBuilder(); 


    private DateTime targetTime;
    private DateTime startTime;
    private bool reminderSet = false;
    private double totalSeconds; 


    private void Start()
    {
        if (setButton != null)
            setButton.onClick.AddListener(SetWateringReminder);

        if (reminderUI != null)
            reminderUI.SetActive(false);

        if (wateringSlider != null)
            wateringSlider.value = 1f;
        if (countdownText != null)
            countdownText.text = ": --";
        //new
       // if (wateredButton != null)
          //  wateredButton.onClick.AddListener(OnWateredButtonClicked);
        //new
        //if (togglePanelButton != null)
           // togglePanelButton.onClick.AddListener(OnTogglePanelClicked);
    }

    private void Update()
    {
        if (reminderSet)
        {
            var remaining = targetTime - DateTime.Now;

            //  UI: "due in: Xh Ym Zs"
            if (countdownText != null)
            {
                int h = remaining.Hours + remaining.Days * 24;
                int m = remaining.Minutes;
                int s = remaining.Seconds;
                countdownText.text = $": {h:D2}h {m:D2}m {s:D2}s";
            }

            if (remaining.TotalSeconds <= 0)
            {
                ShowReminder();
                reminderSet = false;

                if (wateringSlider != null)
                    wateringSlider.value = 0f;

                if (countdownText != null)
                    countdownText.text = "Time to water!";
            }
            else
            {
                if (wateringSlider != null)
                {
                    double elapsed = (DateTime.Now - startTime).TotalSeconds;
                    wateringSlider.value = Mathf.Clamp01((float)((totalSeconds - elapsed) / totalSeconds));
                }
            }
        }
    }

    private void SetWateringReminder()
    {
        int days = int.Parse(daysInput.text);
        int hours = int.Parse(hoursInput.text);
        int minutes = int.Parse(minutesInput.text);

        TimeSpan duration = new TimeSpan(days, hours, minutes, 0);
        totalSeconds = duration.TotalSeconds;

        startTime = DateTime.Now;
        targetTime = startTime + duration;

        reminderSet = true;

        if (wateringSlider != null)
            wateringSlider.value = 1f;

        Debug.Log($"Watering reminder set for: {targetTime} (Duration: {totalSeconds} seconds)");
    }

    private void ShowReminder()
    {
        if (reminderUI != null)
            reminderUI.SetActive(true);

        if (alertAudio != null)
            alertAudio.Play();
    }
    public void OnWateredButtonClicked()
    {
        
        DateTime now = DateTime.Now;
        string timestamp = $"{now:yyyy-MM-dd  HH:mm:ss  (dddd)}";
       // wateringHistory.AppendLine($"✓ Watered on: {timestamp}");

       
       // if (historyText != null)
           // historyText.text = wateringHistory.ToString();
        if (historyItemPrefab != null && historyPanel != null)
        {
            GameObject newItem = Instantiate(historyItemPrefab, historyPanel);
            WateringHistoryItem item = newItem.GetComponent<WateringHistoryItem>();

            if (item != null)
                item.SetText($"✓ Watered on: {timestamp}");

        }

        
        reminderSet = false;
        countdownText.text = ": --";
        wateringSlider.value = 1f;

        
        if (reminderUI != null)
            reminderUI.SetActive(false);
        if (daysInput != null) daysInput.text = "";
        if (hoursInput != null) hoursInput.text = "";
        if (minutesInput != null) minutesInput.text = "";

        if (GamificationManager.Instance != null)
        {
            GamificationManager.Instance.OnWatered();
        }
    }
    public void OnTogglePanelClicked()
    {
        panelVisible = !panelVisible;
        if (targetPanel != null)
            targetPanel.SetActive(panelVisible);
    }

}