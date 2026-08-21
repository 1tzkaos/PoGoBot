# pinch.dex

A two-pointer pinch, injected the way scrcpy injects input: `app_process` running as the
shell user, calling `InputManager.injectInputEvent` by reflection. No root, no APK install,
no accessibility service.

## Why this exists

Pokemon GO's map zoom is a pinch, and `adb shell input` cannot pinch: its CLI only ever
builds a single pointer (`swipe` and `motionevent` both take one x/y). Everything else was
measured and ruled out on the device:

| approach | result |
|---|---|
| `input tap` + `input swipe` (double-tap-drag) | no zoom, at any distance or duration |
| double-tap-**hold**-drag via `input motionevent` | no zoom |
| `input touchscreen swipe` (the v1 qualifier) | no change |
| two concurrent `input swipe` processes | read as two separate gestures |
| `sendevent` on `/dev/input/eventN` | `Permission denied` |
| `KEYCODE_ZOOM_OUT` (169), trackball roll | no effect |

`sendevent` fails even though `shell` is in group `input` and the node is `crw-rw---- root
input`: DAC allows it, SELinux does not (`u:r:shell:s0` writing `u:object_r:input_device:s0`
on a `user` build with `ro.secure=1`, no `su`, `adb root` refused).

Injecting through the framework sidesteps the device node entirely, and the shell uid
already holds `INJECT_EVENTS` - which is exactly why scrcpy can drive the phone.

## Rebuilding

    javac -source 8 -target 8 -bootclasspath $ANDROID_HOME/platforms/android-36/android.jar \
          -cp $ANDROID_HOME/platforms/android-36/android.jar -d classes Pinch.java
    d8 --min-api 31 --output . classes/pinch/Pinch.class
    mv classes.dex ../../pogobot/vendor/pinch.dex

## Arguments

    app_process / pinch.Pinch <cx> <cy> <startGap> <endGap> <steps> <durationMs>

All in device pixels. `startGap > endGap` brings the fingers together, which zooms OUT.
