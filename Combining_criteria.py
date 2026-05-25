import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# thresholds = np.array([1e-10,1e-09,1e-08,1e-07,1e-06,1e-05,1e-04,1e-03])
# for threshold in thresholds:
# # # 1. Load your existing PNG images
#     img3 = Image.open(f'/home/karengarcia/criteria_testing/Figures/Ice_mass_thresholds/CanAM_ice_above_trop_{threshold}kg_test.png')
#     img2 = Image.open(f'/home/karengarcia/criteria_testing/Figures/Ice_mass_thresholds/ERA5_ice_above_trop_{threshold}kg_test.png')
#     img1 = Image.open(f'/home/karengarcia/criteria_testing/Figures/Ice_mass_thresholds/MERRA_ice_above_trop_{threshold}kg_test.png')

#     # # 2. Create a subplot grid (1 row, 3 columns)
#     fig, axes = plt.subplots(3,1, figsize=(15, 5))

#     # 3. Display each image in its respective subplot slot
#     axes[0].imshow(img1)
#     axes[0].axis('off')  # Hide the pixel grid lines/axes

#     axes[1].imshow(img2)
#     axes[1].axis('off')

#     axes[2].imshow(img3)
#     axes[2].axis('off')

#     # 4. Clean up spacing and save the combined image
#     plt.tight_layout()
#     final_plot_path = f'/home/karengarcia/MSc_project/Figures/combined_threshold_cloud_mass_plots_{threshold}.png'
#     plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved to: {final_plot_path}")


#     img4 = Image.open(f'/home/karengarcia/criteria_testing/Figures/Mass_flux_thresholds/MERRA_mf_above_trop_{threshold}kgm2s_test.png')
#     img5 = Image.open(f'/home/karengarcia/criteria_testing/Figures/Mass_flux_thresholds/CanAM_DMCU_above_trop_{threshold}kgm2s_test.png')

#     fig, axes = plt.subplots(2,1, figsize=(15, 5))

#     # 3. Display each image in its respective subplot slot
#     axes[0].imshow(img4)
#     axes[0].axis('off')  # Hide the pixel grid lines/axes

#     axes[1].imshow(img)
#     axes[1].axis('off')

#     # 4. Clean up spacing and save the combined image
#     plt.tight_layout()
#     final_plot_path = f'/home/karengarcia/MSc_project/Figures/combined_threshold_mass_flux_plots_{threshold}.png'
#     plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
#     print(f"Figure saved to: {final_plot_path}")


#Histograms
img6 = Image.open('/home/karengarcia/MSc_project/Figures/MERRA_ice_histogram.png')
img7 = Image.open('/home/karengarcia/MSc_project/Figures/ERA5_ice_histogram.png')
img8 = Image.open('/home/karengarcia/MSc_project/Figures/total_water_histogram.png')

fig, axes = plt.subplots(3, 1, figsize=(15, 5), gridspec_kw={'height_ratios': [1, 1, 0.9]})
axes[0].imshow(img6)
axes[0].axis('off') 

axes[1].imshow(img7)
axes[1].axis('off')

axes[2].imshow(img8)
axes[2].axis('off')

plt.tight_layout()
final_plot_path = '/home/karengarcia/MSc_project/Figures/combined_threshold_cloud_mass_histograms.png'
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {final_plot_path}")


img9  = Image.open('/home/karengarcia/MSc_project/Figures/MERRA_mf_histogram.png')
img10 = Image.open('/home/karengarcia/MSc_project/Figures/DMCU_histogram.png')

fig, axes = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 0.9]})

axes[0].imshow(img9)
axes[0].axis('off') 

axes[1].imshow(img10)
axes[1].axis('off')

plt.tight_layout()
final_plot_path = '/home/karengarcia/MSc_project/Figures/combined_threshold_mf_histograms.png'
plt.savefig(final_plot_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {final_plot_path}")

