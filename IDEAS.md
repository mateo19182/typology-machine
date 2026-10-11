# Ideas parked for later

## HorizonRef
Perspective reference for artists. The user drags a horizon line and 2–3 vanishing points, and the site
renders a real street photo with exactly that perspective.
- Level 360° panos (Mapillary SfM rotation, or GeoCalib on crops). Then any view can be rendered with a
  known camera, so labels come for free.
- Get each pano's Manhattan frame from line segments (DeepLSD) or the OSM street bearing.
- Camera from the drawing: roll = horizon tilt; f² = −(x₁x₂ + y₁y₂) for the two horizontal VPs (relative
  to the image centre); tan(pitch) = y_h / f; yaw from VP position along the horizon.
- Rendering pano to perspective is a small WebGL shader. Add CLIP/SigLIP text filters ("narrow alley").
- Shares the pano, leveling and VP code with the Typology rectification step.

## Others from the brainstorm
LightRef (search portraits by light direction), HandRef (webcam hand pose to reference photos),
CritterRef (animal pose reference), ThumbRef (composition-sketch search), Decade (guess-the-year game vs
a model), Sound Mime (vocal imitation to Freesound).
