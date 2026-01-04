
export const MODEL_TYPES = [
    { label: 'SD1.5', value: 'STABLE_DIFFUSION_15' },
    { label: 'SD1.5 Inpainting', value: 'STABLE_DIFFUSION_15_INPAINTING' },
    { label: 'SD2.0', value: 'STABLE_DIFFUSION_20' },
    { label: 'SD2.0 Inpainting', value: 'STABLE_DIFFUSION_20_INPAINTING' },
    { label: 'SD2.1', value: 'STABLE_DIFFUSION_21' },
    { label: 'SD3', value: 'STABLE_DIFFUSION_3' },
    { label: 'SD3.5', value: 'STABLE_DIFFUSION_35' },
    { label: 'SDXL', value: 'STABLE_DIFFUSION_XL_10_BASE' },
    { label: 'SDXL Inpainting', value: 'STABLE_DIFFUSION_XL_10_BASE_INPAINTING' },
    { label: 'Wuerstchen v2', value: 'WUERSTCHEN_2' },
    { label: 'Stable Cascade', value: 'STABLE_CASCADE_1' },
    { label: 'PixArt Alpha', value: 'PIXART_ALPHA' },
    { label: 'PixArt Sigma', value: 'PIXART_SIGMA' },
    { label: 'Flux Dev', value: 'FLUX_DEV_1' },
    { label: 'Flux Fill Dev', value: 'FLUX_FILL_DEV_1' },
    { label: 'Sana', value: 'SANA' },
    { label: 'Hunyuan Video', value: 'HUNYUAN_VIDEO' },
    { label: 'HiDream Full', value: 'HI_DREAM_FULL' },
    { label: 'Chroma1', value: 'CHROMA_1' },
    { label: 'QwenImage', value: 'QWEN' },
    { label: 'Qwen-Edit', value: 'QWEN_IMAGE_EDIT' },
    { label: 'Kandinsky 5', value: 'KANDINSKY_5' },
    { label: 'Kandinsky 5 Video', value: 'KANDINSKY_5_VIDEO' },
    { label: 'Z-Image', value: 'Z_IMAGE' },
    { label: 'Wan 2.1', value: 'WAN_2_1' },
];

export const getTrainingMethods = (modelType: string) => {
    const sd15Types = ['STABLE_DIFFUSION_15', 'STABLE_DIFFUSION_15_INPAINTING', 'STABLE_DIFFUSION_20', 'STABLE_DIFFUSION_20_INPAINTING', 'STABLE_DIFFUSION_21'];
    const noVaeTypes = ['STABLE_DIFFUSION_3', 'STABLE_DIFFUSION_35', 'STABLE_DIFFUSION_XL_10_BASE', 'STABLE_DIFFUSION_XL_10_BASE_INPAINTING', 'WUERSTCHEN_2', 'STABLE_CASCADE_1', 'PIXART_ALPHA', 'PIXART_SIGMA', 'FLUX_DEV_1', 'FLUX_FILL_DEV_1', 'SANA', 'HUNYUAN_VIDEO', 'HI_DREAM_FULL', 'CHROMA_1'];
    const noEmbeddingTypes = ['QWEN', 'QWEN_IMAGE_EDIT', 'KANDINSKY_5', 'KANDINSKY_5_VIDEO', 'Z_IMAGE', 'WAN_2_1'];

    if (sd15Types.includes(modelType)) {
        return [
            { label: 'Fine Tune', value: 'FINE_TUNE' },
            { label: 'LoRA', value: 'LORA' },
            { label: 'Embedding', value: 'EMBEDDING' },
            { label: 'Fine Tune VAE', value: 'FINE_TUNE_VAE' },
        ];
    } else if (noVaeTypes.includes(modelType)) {
        return [
            { label: 'Fine Tune', value: 'FINE_TUNE' },
            { label: 'LoRA', value: 'LORA' },
            { label: 'Embedding', value: 'EMBEDDING' },
        ];
    } else if (noEmbeddingTypes.includes(modelType)) {
        return [
            { label: 'Fine Tune', value: 'FINE_TUNE' },
            { label: 'LoRA', value: 'LORA' },
        ];
    }
    return [
        { label: 'Fine Tune', value: 'FINE_TUNE' },
        { label: 'LoRA', value: 'LORA' },
    ];
};
