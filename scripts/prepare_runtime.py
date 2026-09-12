"""Small Windows build adaptation; no gameplay or neural changes."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
runtime=ROOT/'.venv/Lib/site-packages/craftground_runtime_mc121'
p=runtime/'src/main/cpp/CMakeLists.txt'
s=p.read_text()
marker='# Flycraft Windows portable build'
if marker not in s:
    s+='\n'+marker+'\nset_target_properties(native-lib PROPERTIES PREFIX "" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release")\nif(MINGW)\n  target_link_options(native-lib PRIVATE -static)\nendif()\n'
    p.write_text(s)
p=ROOT/'.venv/Lib/site-packages/craftground/environment/socket_ipc.py'
s=p.read_text()
needle='    def remove_orphan_java_processes(self):  # noqa: C901\n'
if '# Flycraft: Unix socket cleanup has no role on Windows' not in s:
    assert needle in s
    s=s.replace(needle,needle+'        # Flycraft: Unix socket cleanup has no role on Windows\n        if os.name == "nt":\n            return\n')
    p.write_text(s)
p=ROOT/'.venv/Lib/site-packages/craftground/environment/environment.py'
s=p.read_text()
needle='        pid = p.pid\n'
if '# Flycraft: allow protocol exit to finish saving/cleanup' not in s:
    assert needle in s
    s=s.replace(needle,'        # Flycraft: allow protocol exit to finish saving/cleanup\n        try:\n            p.wait(timeout=10)\n            self.process = None\n            return\n        except subprocess.TimeoutExpired:\n            pass\n\n'+needle)
    p.write_text(s)
p=runtime/'src/main/java/com/kyhsgeekcode/minecraftenv/MinecraftEnv.kt'
s=p.read_text()
start=s.find('            // remove the world file')
end=s.find('            exitProcess(0)',start)+len('            exitProcess(0)')
if '// Flycraft: shut down on the Minecraft client thread' not in s:
    s=s[:start]+'            // Flycraft: shut down on the Minecraft client thread\n            client.scheduleStop()\n            return true'+s[end:]
    p.write_text(s)

p=runtime/'src/main/java/com/kyhsgeekcode/minecraftenv/MinecraftEnv.kt'
s=p.read_text()
needle='            // Flycraft: shut down on the Minecraft client thread\n            client.scheduleStop()'
replacement='            // Flycraft: shut down on the Minecraft client thread\n            ioPhase = IOPhase.SENT_OBSERVATION_SHOULD_READ_ACTION\n            skipSync = true\n            client.scheduleStop()'
if needle in s: p.write_text(s.replace(needle,replacement))


# Keep the real Minecraft client window at the same aspect ratio/resolution as
# the framebuffer delivered to the fly. CraftGround normally launches the
# client at Minecraft's default window size (typically 854x480, 16:9) even
# when imageSizeX/imageSizeY are 640x480 (4:3). Our visible-framebuffer patch
# then stretches the 4:3 framebuffer to that 16:9 window.
#
# CraftGround already contains an update_override_resolutions() helper, but its
# call is disabled upstream. Re-enable it and also pass --width/--height on the
# runClient command so the very first launch (before options.txt exists) is 4:3.
p=ROOT/'.venv/Lib/site-packages/craftground/environment/environment.py'
s=p.read_text()
old = """        if options_txt_path is not None:\n            if os.path.exists(options_txt_path):\n                pass\n                # self.update_override_resolutions(options_txt_path)\n"""
new = """        if options_txt_path is not None:\n            if os.path.exists(options_txt_path):\n                # Flycraft: keep visible Minecraft window matched to capture aspect ratio.\n                self.update_override_resolutions(options_txt_path)\n"""
if old in s:
    s=s.replace(old,new)
elif '# Flycraft: keep visible Minecraft window matched to capture aspect ratio.' not in s:
    needle='                # self.update_override_resolutions(options_txt_path)'
    if needle in s:
        s=s.replace('                pass\\n'+needle,
                    '                # Flycraft: keep visible Minecraft window matched to capture aspect ratio.\\n'
                    '                self.update_override_resolutions(options_txt_path)')

win_old='            cmd = f".\\\\gradlew runClient -w --no-daemon"'
win_new='            cmd = f\'.\\\\gradlew runClient -w --no-daemon --args="--width {self.initial_env.imageSizeX} --height {self.initial_env.imageSizeY}"\''
if win_old in s:
    s=s.replace(win_old,win_new)
unix_old='            cmd = f"./gradlew runClient -w --no-daemon" # --args="--width {self.initial_env.imageSizeX} --height {self.initial_env.imageSizeY}"\''
unix_new='            cmd = f\'./gradlew runClient -w --no-daemon --args="--width {self.initial_env.imageSizeX} --height {self.initial_env.imageSizeY}"\''
if unix_old in s:
    s=s.replace(unix_old,unix_new)
p.write_text(s)

# Present the actual framebuffer instead of leaving a white Windows client area.
p=runtime/'src/main/java/com/kyhsgeekcode/minecraftenv/mixin/RenderMixin.java'
s=p.read_text()
if '// Flycraft visible framebuffer presentation' not in s:
    s=s.replace('private void frameBufferEndWrite(Framebuffer instance) {\n    // do nothing',
                'private void frameBufferEndWrite(Framebuffer instance) {\n    instance.endWrite();')
    s=s.replace('private void frameBufferDraw(Framebuffer instance, int width, int height) {\n    // do nothing',
                'private void frameBufferDraw(Framebuffer instance, int width, int height) {\n    instance.draw(width, height);')
    start=s.index('    RenderSystemPollEventsInvoker.pollEvents();',s.index('private void windowSwapBuffers'))
    end=s.index('\n  }',start)
    s=s[:start]+'    // Flycraft visible framebuffer presentation\n    instance.swapBuffers();'+s[end:]
    p.write_text(s)
