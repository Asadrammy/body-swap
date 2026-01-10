"""
🔧 FORCE MODEL LOADING AT STARTUP
==================================
Add this to main.py startup event
"""
# Add this to src/api/main.py after app creation:

@app.on_event("startup")
async def startup_event():
    """Pre-load models at startup"""
    logger.info("=" * 80)
    logger.info("🚀 PRE-LOADING MODELS AT STARTUP")
    logger.info("=" * 80)
    
    try:
        from ..models.generator import Generator
        import torch
        
        logger.info("Loading Stable Diffusion models...")
        logger.info(f"GPU Available: {torch.cuda.is_available()}")
        
        # Create generator and force load
        generator = Generator()
        logger.info(f"Generator device: {generator.device}")
        
        # Force model loading
        if generator.inpaint_pipe is None:
            logger.info("Models not loaded, initializing...")
            generator._init_inpaint_pipeline()
        
        if generator.inpaint_pipe is not None:
            logger.info("✅ Models loaded successfully!")
            if torch.cuda.is_available():
                mem = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"✅ GPU Memory: {mem:.2f} GB")
                
                # Verify device
                device_str = str(next(generator.inpaint_pipe.unet.parameters()).device)
                logger.info(f"✅ Models on: {device_str}")
        else:
            logger.error("❌ Models failed to load!")
            
    except Exception as e:
        logger.error(f"Error pre-loading models: {e}")
        import traceback
        traceback.print_exc()

