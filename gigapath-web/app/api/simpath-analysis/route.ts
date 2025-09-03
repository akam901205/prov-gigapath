import { NextRequest, NextResponse } from 'next/server'

// Configure route for long-running operations - same as true-tiered
export const maxDuration = 900 // 15 minutes
export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    // Handle FormData EXACTLY like true-tiered analysis
    const formData = await request.formData()
    const image = formData.get('image') as File

    if (!image) {
      return NextResponse.json({ error: 'No image provided' }, { status: 400 })
    }

    console.log(`SimPath direct analysis: ${image.name}`)
    console.log(`Image size: ${image.size} bytes`)

    // Convert image to base64 server-side - same as true-tiered
    const imageBuffer = await image.arrayBuffer()
    const imageBase64 = Buffer.from(imageBuffer).toString('base64')

    // Create request payload for RunPod API
    const requestPayload = {
      input: {
        image_base64: imageBase64,
        metrics: ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'braycurtis', 'canberra', 'seuclidean', 'pearson']
      }
    }

    // Direct call to RunPod API
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 900000) // 15 minutes

    console.log('🌐 Calling RunPod SimPath API directly...')
    const response = await fetch('https://8v9wob2mln55to-8007.proxy.runpod.net/api/simpath-analysis', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'NextJS-Direct/1.0'
      },
      body: JSON.stringify(requestPayload),
      signal: controller.signal
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`RunPod API error ${response.status}: ${errorText}`)
      return NextResponse.json(
        { error: 'SimPath analysis failed', details: `Status ${response.status}` },
        { status: response.status }
      )
    }

    const result = await response.json()
    console.log('✅ SimPath analysis completed successfully')
    
    return NextResponse.json(result)

  } catch (error: any) {
    console.error('SimPath direct API error:', error)
    return NextResponse.json(
      { error: 'SimPath analysis failed', details: error.message },
      { status: 500 }
    )
  }
}