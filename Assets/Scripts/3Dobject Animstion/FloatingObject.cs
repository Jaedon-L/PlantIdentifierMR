using UnityEngine;
using DG.Tweening;

public class FloatingObject : MonoBehaviour
{
    private Tween floatTween;
    public AudioSource audioSource; // صدا (لوپ‌دار)
    public ParticleSystem particles; // پارتیکل سیستم

    void Start()
    {
        StartFloating();
    }

    public void StartFloating()
    {
        floatTween = transform.DOMoveY(transform.position.y + 0.2f, 1f)
                              .SetLoops(-1, LoopType.Yoyo)
                              .SetEase(Ease.InOutSine);

        // شروع صدا و پارتیکل
        if (audioSource != null && !audioSource.isPlaying)
            audioSource.Play();

        if (particles != null && !particles.isPlaying)
            particles.Play();
    }

    public void StopFloating()
    {
        if (floatTween != null && floatTween.IsActive())
            floatTween.Kill();

        if (audioSource != null)
            audioSource.Stop();

        if (particles != null)
            particles.Stop();
    }

    public void ResumeFloating()
    {
        StopFloating(); // برای اطمینان
        StartFloating();
    }
}
