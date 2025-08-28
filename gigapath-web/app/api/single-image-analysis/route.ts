import { NextRequest, NextResponse } from 'next/server'

const SINGLE_IMAGE_API_URL = process.env.GIGAPATH_SINGLE_IMAGE_API_URL || 'http://localhost:8001'

// Configure route for long-running operations
export const maxDuration = 900 // 15 minutes
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const image = formData.get('image') as File

    if (!image) {
      return NextResponse.json(
        { error: 'No image provided' },
        { status: 400 }
      )
    }

    console.log(`Analyzing single image: ${image.name}`)

    // Convert image to base64 for backend API
    const imageBuffer = await image.arrayBuffer()
    const imageBase64 = Buffer.from(imageBuffer).toString('base64')

    // Create JSON payload as expected by fast_api.py
    const requestPayload = {
      input: {
        image_base64: imageBase64,
        encoder_type: "tile",
        mode: "inference"
      }
    }

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 900000) // 15 minutes
    
    const response = await fetch(`${SINGLE_IMAGE_API_URL}/api/single-image-analysis`, {
      method: 'POST',
      body: JSON.stringify(requestPayload),
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal,
      keepalive: true
    })
    
    clearTimeout(timeoutId)

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Single image analysis error:', errorText)
      return NextResponse.json(
        { error: 'Analysis failed', details: errorText },
        { status: response.status }
      )
    }

    const analysisData = await response.json()
    
    console.log('Single image analysis completed successfully')
    console.log('Backend response keys:', Object.keys(analysisData))
    console.log('Has gigapath_verdict:', 'gigapath_verdict' in analysisData)
    if (analysisData.gigapath_verdict) {
      console.log('GigaPath verdict keys:', Object.keys(analysisData.gigapath_verdict))
    }
    return NextResponse.json(analysisData)

  } catch (error) {
    console.error('Single image analysis proxy error:', error)
    return NextResponse.json(
      { error: 'Analysis failed', details: error.message },
      { status: 500 }
    )
  }
}