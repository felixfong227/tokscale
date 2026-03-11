import { NextResponse } from "next/server";
import { db, deviceCodes, users } from "@/lib/db";
import { eq, and, gt } from "drizzle-orm";
import { issuePersonalToken } from "@/lib/auth/personalTokens";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { deviceCode } = body;

    if (!deviceCode) {
      return NextResponse.json(
        { error: "Missing device code" },
        { status: 400 }
      );
    }

    // Find the device code record
    const [record] = await db
      .select()
      .from(deviceCodes)
      .where(
        and(
          eq(deviceCodes.deviceCode, deviceCode),
          gt(deviceCodes.expiresAt, new Date())
        )
      )
      .limit(1);

    if (!record) {
      return NextResponse.json({ status: "expired" });
    }

    // Check if user has authorized
    if (!record.userId) {
      return NextResponse.json({ status: "pending" });
    }

    // User has authorized - create API token
    const [user] = await db
      .select()
      .from(users)
      .where(eq(users.id, record.userId))
      .limit(1);

    if (!user) {
      return NextResponse.json(
        { error: "User not found" },
        { status: 500 }
      );
    }

    const issuedToken = await issuePersonalToken({
      userId: user.id,
      name: record.deviceName || "CLI",
      ensureUniqueName: true,
    });

    // Delete the device code (one-time use)
    await db.delete(deviceCodes).where(eq(deviceCodes.id, record.id));

    return NextResponse.json({
      status: "complete",
      token: issuedToken.token,
      user: {
        username: user.username,
        avatarUrl: user.avatarUrl,
      },
    });
  } catch (error) {
    console.error("Device poll error:", error);
    return NextResponse.json(
      { error: "Failed to poll device code" },
      { status: 500 }
    );
  }
}
