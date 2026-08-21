package pinch;

import android.view.InputDevice;
import android.view.InputEvent;
import android.view.MotionEvent;
import java.lang.reflect.Method;

/**
 * Two-pointer pinch, injected the way scrcpy does it: app_process as the shell user,
 * InputManager.injectInputEvent by reflection. No root, no APK install.
 *
 * adb shell requires INJECT_EVENTS, which the shell uid holds - the same grant scrcpy
 * relies on. `input` cannot do this because its CLI only ever builds one pointer.
 */
public final class Pinch {

    private static Object manager;
    private static Method injector;

    public static void main(String[] args) throws Exception {
        float cx = Float.parseFloat(args[0]);
        float cy = Float.parseFloat(args[1]);
        float startGap = Float.parseFloat(args[2]);
        float endGap = Float.parseFloat(args[3]);
        int steps = Integer.parseInt(args[4]);
        int durationMs = Integer.parseInt(args[5]);

        Class<?> cls;
        try {
            cls = Class.forName("android.hardware.input.InputManager");
            manager = cls.getMethod("getInstance").invoke(null);
        } catch (Throwable t) {
            cls = Class.forName("android.hardware.input.InputManagerGlobal");
            manager = cls.getMethod("getInstance").invoke(null);
        }
        injector = cls.getMethod("injectInputEvent", InputEvent.class, int.class);

        long down = android.os.SystemClock.uptimeMillis();
        send(down, down, MotionEvent.ACTION_DOWN, 1, cx, cy, startGap);
        send(down, android.os.SystemClock.uptimeMillis(),
             MotionEvent.ACTION_POINTER_DOWN | (1 << MotionEvent.ACTION_POINTER_INDEX_SHIFT),
             2, cx, cy, startGap);

        int sleep = Math.max(1, durationMs / Math.max(1, steps));
        for (int i = 1; i <= steps; i++) {
            Thread.sleep(sleep);
            float gap = startGap + (endGap - startGap) * i / steps;
            send(down, android.os.SystemClock.uptimeMillis(),
                 MotionEvent.ACTION_MOVE, 2, cx, cy, gap);
        }

        send(down, android.os.SystemClock.uptimeMillis(),
             MotionEvent.ACTION_POINTER_UP | (1 << MotionEvent.ACTION_POINTER_INDEX_SHIFT),
             2, cx, cy, endGap);
        send(down, android.os.SystemClock.uptimeMillis(),
             MotionEvent.ACTION_UP, 1, cx, cy, endGap);
        System.out.println("pinch sent: gap " + startGap + " -> " + endGap);
    }

    /** Two fingers on a vertical line through (cx, cy), `gap` apart. */
    private static void send(long downTime, long eventTime, int action, int count,
                             float cx, float cy, float gap) throws Exception {
        MotionEvent.PointerProperties[] props = new MotionEvent.PointerProperties[count];
        MotionEvent.PointerCoords[] coords = new MotionEvent.PointerCoords[count];
        for (int i = 0; i < count; i++) {
            MotionEvent.PointerProperties p = new MotionEvent.PointerProperties();
            p.id = i;
            p.toolType = MotionEvent.TOOL_TYPE_FINGER;
            props[i] = p;
            MotionEvent.PointerCoords c = new MotionEvent.PointerCoords();
            c.x = cx;
            c.y = cy + (i == 0 ? -gap / 2f : gap / 2f);
            c.pressure = 1f;
            c.size = 1f;
            coords[i] = c;
        }
        MotionEvent ev = MotionEvent.obtain(downTime, eventTime, action, count, props, coords,
                0, 0, 1f, 1f, 0, 0, InputDevice.SOURCE_TOUCHSCREEN, 0);
        injector.invoke(manager, ev, 2 /* INJECT_INPUT_EVENT_MODE_WAIT_FOR_FINISH */);
        ev.recycle();
    }
}
