---
created: 2025-07-12T22:25:37 (UTC -04:00)
tags: []
source: moz-extension://d426f292-5da2-492a-83b1-13c3507c7a2f/home
author: 
---

# PACE - External - Phoenix Cluster Resources

> ## Excerpt
> Knowledge Article

---
Skip to page content

Knowledge Article

## Detailed Node Specs[¶](moz-extension://d426f292-5da2-492a-83b1-13c3507c7a2f/home#detailed-node-specs "Permanent link")

-   Most nodes includes the following common features:  
    -   Dual Intel Xeon Gold 6226 CPUs @ 2.7 GHz (24 cores/node)
    -   DDR4-2933 MHz DRAM
    -   Infiniband 100HDR interconnect
-   4 **gpu-h100** nodes include the following common features (1 DGX added November 2023; 3 HGX added January 2024)
    -   1 DGX Node:
        -   Intel Xeon Platinum 8480CL CPUs @ 2.0 GHz (112 cores/node)
        -   8x Nvidia Tensor Core H100 80GB
        -   **Note: Only available using the embers QoS for most Phoenix users**
    -   3 HGX Nodes:
        -   Dual Intel Xeon Platinum 8462Y+ CPUs @ 2.8 GHz (64 cores/node)
        -   2TB DDR5 DRAM
        -   4TB NVMe
        -   100 Gbps InfiniBand NICs
        -   8x Nvidia Tensor Core H100 80GB
-   6 **gpu-h200** nodes added Jan 2025, including the following common features:
    -   2x INTEL(R) XEON(R) PLATINUM 8562Y (32c, 2.8GHz) - 64 cores/node
    -   2TB DDR5-5600MHz DRAM
    -   4x200 Gbps InfiniBand NICs
    -   28TB Local NVMe
    -   8x NVIDIA Tensor Core H200-HGX 142GB
-   10 **gpu-l40s** nodes include the following common features (8 added August 30 2024)
    -   Dual Intel Xeon 6426Y CPUs @ 2.5 GHz (32 cores/node)
    -   512GB DDR5 RAM
    -   61.44TB NVMe
    -   100 Gbps InfiniBand NICs
    -   8x Nvidia L40S 48GB GPUs
    -   **Note: 8 of these are only available using the embers QoS for most Phoenix users, 2 are available through Inferno  
        **
-   4 **cpu-amd** nodes include the following common features (4 added November 2023)
    -   Dual AMD Epyc 9534 CPUs @ 2.45 GHz (128 cores/node)
    -   3TB DDR5 DRAM
    -   4TB NVMe
    -   100 Gbps InfiniBand NICs
-   40 **cpu-large** nodes have been added on February 21, 2023, with Dual Intel Xeon Gold 6226R CPUs @ 2.9 GHz (32 cores/node) and 768 GB of RAM
-   8 **cpu-amd** nodes include the following common features (4 added November 7, 2022; 4 added February 23, 2023):  
    -   Dual AMD Epyc 7713 CPUs @ 2.0 GHz (128 cores/node)
    -   512GB DDR4 DRAM
    -   1.6TB NVMe
-   12 **gpu-a100** nodes include the following common features (5 added November 7, 2022; 6 added Feburary 23, 2023, 1 added February 28, 2023):  
    -   Dual AMD Epyc 7513 CPUs @ 2.6 GHz (64 cores/node)
    -   512GB DDR4 DRAM
    -   2x Nvidia Tensor Core A100 40GB (6 nodes) or 80GB (6 nodes) GPUs
    -   1.6TB NVMe
-   1 additional **gpu-a100** node includes the following common features (added April 1, 2024):  
    -   Dual AMD Epyc 7543 CPUs @ 2.8 GHz (64 cores/node)
    -   2TB DDR4 DRAM
    -   8x Nvidia Tensor Core A100 80GB GPUs
    -   180TB NVMe
    -   **Note: Only available using the embers QoS for most Phoenix users**
-   1 **cpu-pmem** node (added on April 11, 2023) has a large amount of memory (1.5 TB). This memory is composed of 192 GB of DDR4-2933 ECC-DRAM and 1.3125 TB of 2666MHz DCPMM (Intel Optane persistent memory). It has 24 cores (Dual Intel Xeon Gold 6226 CPUs @ 2.7 GHz) and Infiniband 100HDR interconnect.
-   The following chart provides detailed specifications for the 1393 nodes that were part of the Phoenix-Slurm cluster migration for Phases 1 (October 10-12, 2022) through 6 (January 30-February 3, 2023). Details are also included for additional nodes: 1 cpu-pmem node, 40 cpu-large nodes, 12 cpu-amd nodes, 12 gpu-a100, and 4 gpu-h100 nodes:

| Node Class | Quantity | RAM | Storage | CPU Specs | GPU Specs |
| --- | --- | --- | --- | --- | --- |
| CPU-192GB | 851 | 192 GB | 1.6 TB NVMe storage | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) |   |
| CPU-384GB | 239 | 384 GB | 1.6 TB NVMe storage | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) |   |
| CPU-768GB | 104 | 768 GB | 1.6 TB NVMe storage | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) |   |
| CPU-384GB-SAS | 75 | 192 GB | 8.0 TB SAS storage | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) |   |
| CPU-768GB-SAS | 4 | 384 GB | 8.0 TB SAS storage | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) |   |
| CPU-PMEM | 1 | 1.5 TB | 1.6 TB NVMe storage | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) |   |
| GPU-192GB-V100 | 21 | 192 GB |   | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) | 2x Nvidia Tesla V100 (16GB or 32GB) |
| GPU-384GB-V100 | 27 | 384 GB |   | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) | 2x Nvidia Tesla V100 (16GB or 32GB) |
| GPU-768GB-V100 | 5 | 768 GB |   | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) | 2x Nvidia Tesla V100 (16GB) |
| GPU-384GB-RTX6000 | 32 | 384 GB |   | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) | 4x Nvidia Quadro RTX6000 (24GB) |
| GPU-768GB-RTX6000 | 5 | 768 GB |   | 2x Intel Xeon Gold 6226 @ 2.7GHz (24 cores/node) | 4x Nvidia Quadro RTX6000 (24GB) |
| CPU-512GB-AMD | 8 | 512 GB | 1.6 TB NVMe storage | 2x AMD Epyc 7713 @ 2.0GHz (128 cores/node) |   |
| CPU-3TB-AMD\* | 4 | 3 TB | 4.0 TB NVMe storage | 2x AMD Epyc 9534 @ 2.45GHz (128 cores/node) |   |
| GPU-512GB-A100 | 12 | 512 GB | 1.6 TB NVMe storage | 2x AMD Epyc 7513 @ 2.6GHz (64 cores/node) | 2x Nvidia Tensor Core A100 (40GB or 80GB) |
| GPU-A100-HGX\* | 1 | 2 TB | 180 TB NVMe storage | 2x AMD Epyc 7543 @ 2.8GHz (64 cores/node) | 8x Nvidia Tensor Core A100 (80GB) |
| GPU-H100-HGX | 3 | 2 TB | 4.0 TB NVMe storage | 2x Intel Xeon Platinum 8462Y+ @ 2.8GHz (64 cores/node) | 8x Nvidia Tensor Core H100 (80GB) |
| GPU-H100-DGX\* | 1 | 2 TB |   | 
2x Intel Xeon Platinum 8480CL @ 2.0GHz (112 cores/node)

 | 8x Nvidia Tensor Core H100 (80GB) |
| GPU-H200-HGX | 6 | 2TB | 28.0 TB NVMe storage | 2x Intel Xeon Platinum 8562Y+ @ 2.8GHz (64 cores/node) | 8x Nvidia Tensor Core H200 (142GB) |
| GPU-L40S | 10 | 512 GB | 61.44 TB NVMe storage | 2x Intel Xeon 6426Y @ 2.5GHz (32 cores/node) | 8x Nvidia L40S (48GB) |
|   | 1393\*\* |   |   |   |   |

\*Note: Only available using the embers QoS for most Phoenix users

\*\*Note: Total node count may vary due to maintenance

## Partitions[¶](moz-extension://d426f292-5da2-492a-83b1-13c3507c7a2f/home#partitions "Permanent link")

-   Jobs are assigned to Slurm partitions automatically based on your charge account (internal or external) and the most significant resources requested (gpu, memory requirements, etc).
-   Jobs will only be charged if the inferno QOS is selected.
-   Slurm partitions assigned determine how much users are charged based on current [rates](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042194#rate-study).
-   Slurm partitions include the following node classes and are assigned by the scheduler based on availability:

| Partition | Node Class |
| --- | --- |
| cpu-small | CPU-192GB, CPU-384GB, CPU-384GB-SAS, CPU-768GB, CPU-768GB-SAS |
| cpu-medium | CPU-384GB, CPU-384GB-SAS, CPU-768GB, CPU-768GB-SAS |
| cpu-large | CPU-768GB, CPU-768GB-SAS |
| cpu-sas | CPU-384GB-SAS, CPU-768GB-SAS |
| cpu-pmem | CPU-PMEM |
| gpu-v100 | GPU-192GB-V100, GPU-384GB-V100, GPU-768GB-V100 |
| gpu-rtx6000 | GPU-384GB-RTX6000, GPU-768GB-RTX6000 |
| cpu-amd | CPU-512GB-AMD, CPU-3TB-AMD |
| gpu-a100 | GPU-512GB-A100 |
| gpu-h100 | GPU-H100-DGX, GPU-H100-HGX |
| gpu-l40s | GPU-L40S |

-   The partitions for external users have the same names with "-X" added (i.e. cpu-small-X, cpu-medium-X).

## Job Submit Flowchart[¶](moz-extension://d426f292-5da2-492a-83b1-13c3507c7a2f/home#job-submit-flowchart "Permanent link")

-   When submitting a job to the Slurm scheduler using interactive mode (using `salloc`) or with a script (using `sbatch`), the resources requested will determine the partition assigned, as illustrated in the following flowchart:

![phx](moz-extension://d426f292-5da2-492a-83b1-13c3507c7a2f/jobSubmit-flowchart.pngx)

<table><tbody><tr><td><p><span><strong><img src="moz-extension://d426f292-5da2-492a-83b1-13c3507c7a2f/sys_attachment.do?sys_id=51f0686e1be661d083cfebd7b04bcb4c" alt="alt" width="26" height="26" tabindex="0"></strong></span></p></td><td><p><span><span><strong>Important</strong></span></span></p><p><span><span>The scheduler reserves 8GB of memory for system processes, so the total available memory for jobs on a given node is reduced accordingly.</span></span></p><p><span><span>For memory requests on the border between partitions (i.e. --mem-per-cpu=8GB for cpu-small/cpu-medium, --mem-per-cpu=16GB for cpu-medium/cpu-large), jobs will allocate nodes of the larger partition node type but charge for the smaller partition type.</span></span></p></td></tr></tbody></table>

___

Copy Permalink

### Was this article helpful?

___

#### Related Articles

No content to display

#### ASC Most Viewed Articles

KB0041418 v13.0 Payroll & Taxes

KB0043151 v5.0 Performance Management

KB0041374 v17.0 Performance Management

KB0041688 v11.0 Time Away from Work

KB0041747 v15.0 Reporting Time Worked and Timekeeping

#### ASC Most Useful Articles

KB0043151 v5.0 Performance Management

KB0042087 v5.0 Performance Management

KB0041418 v13.0 Payroll & Taxes

KB0041374 v17.0 Performance Management
