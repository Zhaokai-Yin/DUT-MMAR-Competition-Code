"""
修复版本：添加特征归一化，解决分数下降问题
"""
import sys
import os

# 复制原文件并修复
with open('zero_shot_inference_pose_enhanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修复1: 在MultiBackbonePoseExtractor的forward中添加特征归一化
old_forward = """    def forward(self, rgb_input, depth_input, ir_input):
        \"\"\"
        提取多backbone特征
        注意：骨骼点特征需要在CPU上单独提取
        \"\"\"
        all_features = []
        
        # 提取每个backbone的特征
        for name in self.backbone_names:
            extractor = self.backbone_extractors[name]
            rgb_feat = extractor(rgb_input)
            depth_feat = extractor(depth_input)
            ir_feat = extractor(ir_input)
            # 拼接该backbone的三个模态特征
            backbone_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
            all_features.append(backbone_feat)
        
        # 拼接所有backbone特征
        combined_feat = torch.cat(all_features, dim=1)
        return combined_feat"""

new_forward = """    def forward(self, rgb_input, depth_input, ir_input):
        \"\"\"
        提取多backbone特征
        注意：骨骼点特征需要在CPU上单独提取
        \"\"\"
        all_features = []
        
        # 提取每个backbone的特征
        for name in self.backbone_names:
            extractor = self.backbone_extractors[name]
            rgb_feat = extractor(rgb_input)
            depth_feat = extractor(depth_input)
            ir_feat = extractor(ir_input)
            
            # 对每个backbone的特征进行L2归一化（重要！）
            # 这样可以避免不同backbone特征尺度差异导致的问题
            rgb_feat = F.normalize(rgb_feat, p=2, dim=1)
            depth_feat = F.normalize(depth_feat, p=2, dim=1)
            ir_feat = F.normalize(ir_feat, p=2, dim=1)
            
            # 拼接该backbone的三个模态特征
            backbone_feat = torch.cat([rgb_feat, depth_feat, ir_feat], dim=1)
            all_features.append(backbone_feat)
        
        # 拼接所有backbone特征
        combined_feat = torch.cat(all_features, dim=1)
        return combined_feat"""

if old_forward in content:
    content = content.replace(old_forward, new_forward)
    print("✓ 已添加特征归一化")
else:
    print("⚠️  未找到目标代码，可能需要手动修复")

# 保存修复后的文件
with open('zero_shot_inference_pose_enhanced_fixed.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ 修复文件已保存: zero_shot_inference_pose_enhanced_fixed.py")
print("\n请使用修复后的文件重新运行！")




