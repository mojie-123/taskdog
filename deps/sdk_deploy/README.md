# SDK DEPLOY
## SDK Overview
This repository contains the robotics control SDK, currently supporting Lite3 and M20.

> [!NOTE]
> Damage caused by using SDK is not covered under warranty!

 The repository is structured as follows:
- `src/drdds`: The drdds communication format used by the SDK.
### Lite3
- `src/Lite3_sdk_deploy`: The source code for the Lite3 SDK deploy.
- `src/lite3_sdk_service`: The source code for Lite3 SDK mode switching and frequency modification services.
- `src/lite3_transfer`: The source code for Lite3 UDP-ROS2 message transfer.  

Before using the Lite3 SDK, please refer to the [Lite3 SDK Service Guide](src/lite3_sdk_service/README.md). For the Lite3 SDK deployment process, please refer to the [Lite3 SDK Deployment Guide](src/Lite3_sdk_deploy/README.md).
### M20
- `src/M20_sdk_deploy`: The source code for the M20 SDK deploy.  

For the M20 SDK deployment process, please refer to the [M20 SDK Deployment Guide](/src/M20_sdk_deploy/README.md).  
## Contributors
See the [Contributors](Contributors.md) page for a list of contributors.

## SDK总览
本仓库包含机器人控制SDK，当前支持Lite3和M20平台。

> [!NOTE]
> 因使用 SDK 造成的设备损坏不在保修范围内！

仓库结构如下：
- `src/drdds`：SDK使用的drdds通信格式。
### Lite3
- `src/Lite3_sdk_deploy`：Lite3 SDK部署的源代码。
- `src/lite3_sdk_service`：Lite3 SDK模式切换、频率修改服务的源代码。
- `src/lite3_transfer`：Lite3 UDP-ROS2消息转换的源代码。

使用Lite3 SDK前请查看[Lite3 SDK服务说明](/src/lite3_sdk_service/README.md)，Lite3 SDK部署流程请查看[Lite3 SDK部署说明](src/Lite3_sdk_deploy/README.md)。
### M20
- `src/M20_sdk_deploy`：M20 SDK部署的源代码。

M20 SDK部署流程请查看[M20 SDK部署说明](/src/M20_sdk_deploy/README.md)。
## 贡献者
请参阅[贡献者](Contributors.md)页面查看贡献者列表。
