"""Expose genuine breaking progress through existing integer telemetry transport."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
runtime=ROOT/'.venv/Lib/site-packages/craftground_runtime_mc121'
base=runtime/'src/main/java/com/kyhsgeekcode/minecraftenv'
p=base/'mixin/FlycraftBreakingAccessor.java'
accessor='''package com.kyhsgeekcode.minecraftenv.mixin;
import net.minecraft.client.network.ClientPlayerInteractionManager;
import net.minecraft.util.math.BlockPos;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
@Mixin(ClientPlayerInteractionManager.class)
public interface FlycraftBreakingAccessor {
 @Accessor("currentBreakingProgress") float flycraftProgress();
 @Accessor("currentBreakingPos") BlockPos flycraftPosition();
 @Accessor("breakingBlock") boolean flycraftBreaking();
}
'''
if not p.exists() or p.read_text()!=accessor:p.write_text(accessor)
p=runtime/'src/main/resources/com.kyhsgeekcode.minecraftenv.mixin.json'
data=json.loads(p.read_text())
if 'FlycraftBreakingAccessor' not in data['client']:
    data['client'].append('FlycraftBreakingAccessor');p.write_text(json.dumps(data,indent=2))
p=base/'MinecraftEnv.kt';s=p.read_text()
if '// Flycraft training telemetry' not in s:
    s=s.replace('    private val variableCommandsAfterReset', '    // Flycraft training telemetry: external observer only\n    private var flycraftTarget: BlockPos? = null\n    private val variableCommandsAfterReset')
    s=s.replace('                for (command in commands) {','''                for (command in commands) {
                    if (command.startsWith("flycraft_target ")) {
                        val xyz = command.split(" ").drop(1).map { it.toInt() }
                        flycraftTarget = BlockPos(xyz[0], xyz[1], xyz[2])
                        continue
                    }''')
    s=s.replace('                    image = imageByteString1','''                    // Float bits preserve genuine progress without a protobuf rebuild.
                    flycraftTarget?.let { target ->
                        val manager = client.interactionManager as com.kyhsgeekcode.minecraftenv.mixin.FlycraftBreakingAccessor
                        val progress = if (manager.flycraftBreaking() && manager.flycraftPosition() == target) manager.flycraftProgress() else 0.0f
                        miscStatistics["flycraft.progress_bits"] = java.lang.Float.floatToIntBits(progress)
                        miscStatistics["flycraft.target_present"] = if (world.getBlockState(target).isOf(net.minecraft.block.Blocks.OAK_LOG)) 1 else 0
                        miscStatistics["flycraft.target_x"] = target.x
                        miscStatistics["flycraft.target_y"] = target.y
                        miscStatistics["flycraft.target_z"] = target.z
                    }
                    image = imageByteString1''')
    p.write_text(s)
print('Training telemetry prepared; Gradle will rebuild the changed Java/Kotlin files on launch.')
